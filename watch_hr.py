"""
watch_hr.py — BLE smartwatch heart-rate reader for BREATHE / BioAQI

Reads live heart rate from a smartwatch broadcasting over the standard
Bluetooth LE Heart Rate Profile (e.g. a Wear OS Galaxy Watch running the
"Heart for Bluetooth" broadcaster app). Mirrors the sensor.py design: a
background daemon thread keeps a shared latest-reading dict, and the app
polls get_hr() without blocking.

The watch's HR is used as a *ventilation / exertion* proxy — higher HR means
more air (and thus more pollutant dose) is being inhaled. It is NOT a claim
that a high pulse means pollution is directly harming the user.

Usage:
    import watch_hr
    watch_hr.start()                 # auto-connects to first HR broadcaster
    # or: watch_hr.start(address="AA:BB:CC:DD:EE:FF")  # deterministic connect
    r = watch_hr.get_hr()
    # {"bpm": 78, "ts": 1234.5, "stale": False, "error": None}
"""

import asyncio
import threading
import time

from bleak import BleakScanner, BleakClient

# Standard GATT Heart Rate Service / Measurement characteristic UUIDs
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# A reading older than this (seconds) is considered stale — the UI should show
# "reconnecting" rather than a frozen number that silently corrupts the score.
STALE_AFTER_S = 5.0

# Shared state — written by the background thread, read by Streamlit/the app.
_latest: dict = {"bpm": None, "ts": None, "error": None}
_thread: threading.Thread | None = None
_stop = threading.Event()


# ── HR payload parsing ────────────────────────────────────────────────────────

def parse_heart_rate(data: bytearray) -> int | None:
    """Parse the standard Heart Rate Measurement characteristic (0x2A37).

    Flags bit 0 selects the HR value format: 0 = uint8, 1 = uint16 (LE).
    """
    if not data or len(data) < 2:
        return None
    flags = data[0]
    if flags & 0x01:  # uint16
        return int.from_bytes(data[1:3], byteorder="little")
    return data[1]    # uint8


# ── BLE connect / subscribe loop (runs inside the background thread) ──────────

async def _find_candidates(address: str | None, name_hint: str | None):
    """Return an ordered list of BLE devices worth trying.

    Galaxy watches expose the HR service only AFTER connecting — they do NOT
    advertise 0x180D — so we can't filter by advertised service. We therefore
    gather candidates by (1) explicit address, (2) advertised HR service if any,
    (3) name match (e.g. 'Galaxy Watch'), and let the connect loop verify which
    one actually exposes the HR characteristic. Matching by name also survives
    the watch's rotating (resolvable random) address.
    """
    if address:
        dev = await BleakScanner.find_device_by_address(address, timeout=10.0)
        return [dev] if dev else []

    devices = await BleakScanner.discover(timeout=6.0, return_adv=True)
    prioritized, named = [], []
    for dev, adv in devices.values():
        uuids = [u.lower() for u in (adv.service_uuids or [])]
        name = dev.name or adv.local_name or ""
        if HR_SERVICE_UUID in uuids:
            prioritized.append(dev)
        elif name_hint and name_hint.lower() in name.lower():
            named.append(dev)
    return prioritized + named


def _on_hr(_sender, data: bytearray) -> None:
    """Notification callback — update shared state with the latest HR."""
    global _latest
    bpm = parse_heart_rate(data)
    print(f"[watch_hr] notify raw={bytes(data).hex()} -> bpm={bpm}", flush=True)
    if bpm is not None and bpm > 0:
        _latest = {"bpm": bpm, "ts": time.monotonic(), "error": None}


async def _run(address: str | None, name_hint: str | None) -> None:
    """Connect, subscribe, and auto-reconnect until stop() is called.

    Mirrors the manual-pick script that worked: it does NOT gate on the HR
    characteristic being enumerated — it just subscribes and treats a device as
    the right one if real HR data actually arrives within a few seconds. That
    selects the broadcaster among multiple 'Galaxy Watch' entries.
    """
    global _latest
    while not _stop.is_set():
        try:
            cands = await _find_candidates(address, name_hint)
            if not cands:
                _latest = {"bpm": None, "ts": None,
                           "error": "No watch found — start the broadcaster app on the watch."}
                await asyncio.sleep(2.0)
                continue

            # With an explicit address (from the UI picker / manual choice) we
            # behave exactly like read_watch_hr.py: connect, subscribe, stream —
            # no data-gating. The gate is only for nameless auto-mode, to reject
            # the wrong device among several candidates.
            require_data = address is None

            connected = False
            for dev in cands:
                if _stop.is_set():
                    break
                try:
                    async with BleakClient(dev, timeout=15.0) as client:
                        # Subscribe blindly (like the working script) — don't
                        # trust the advertised/enumerated characteristic list.
                        try:
                            await client.start_notify(HR_MEASUREMENT_CHAR_UUID, _on_hr)
                        except Exception:
                            continue  # can't subscribe here — try next candidate

                        if require_data:
                            # Confirm it's really streaming before committing.
                            mark = time.monotonic()
                            got = False
                            for _ in range(12):  # up to ~6s
                                if _stop.is_set():
                                    break
                                ts = _latest.get("ts")
                                if ts is not None and ts > mark:
                                    got = True
                                    break
                                await asyncio.sleep(0.5)
                            if not got:
                                await client.stop_notify(HR_MEASUREMENT_CHAR_UUID)
                                continue  # subscribed but idle — wrong device

                        connected = True
                        print(f"[watch_hr] connected + subscribed to {dev.address}", flush=True)
                        while not _stop.is_set() and client.is_connected:
                            await asyncio.sleep(0.5)
                        await client.stop_notify(HR_MEASUREMENT_CHAR_UUID)
                        break
                except Exception:
                    continue  # this candidate failed — try the next

            if not connected:
                _latest = {"bpm": None, "ts": None,
                           "error": "Found the watch but no HR stream — start/keep-awake the "
                                    "broadcaster app, grant sensor permission, or pair in Windows."}
                await asyncio.sleep(2.0)

        except Exception as exc:  # noqa: BLE stacks raise many device-specific errors
            _latest = {"bpm": None, "ts": None, "error": str(exc)}
            await asyncio.sleep(2.0)


def _thread_main(address: str | None, name_hint: str | None) -> None:
    """Daemon-thread entry point — owns its own asyncio event loop."""
    asyncio.run(_run(address, name_hint))


# ── Public API (mirrors sensor.py) ────────────────────────────────────────────

def start(address: str | None = None, name_hint: str | None = "Galaxy Watch") -> None:
    """Start the background BLE reader.

    address   : optional exact watch BLE address for deterministic connect.
    name_hint : device-name substring to match when no address is given
                (default 'Galaxy Watch'). Robust to the watch's rotating address.
                Whichever candidate actually exposes the HR characteristic wins.
    """
    global _thread, _latest
    _stop.clear()
    _latest = {"bpm": None, "ts": None, "error": None}
    _thread = threading.Thread(
        target=_thread_main,
        args=(address, name_hint),
        daemon=True,   # exits automatically when the app process exits
        name="watch-hr-reader",
    )
    _thread.start()


def stop() -> None:
    """Signal the background reader to disconnect and exit."""
    _stop.set()


def get_hr() -> dict:
    """Latest HR snapshot (thread-safe): {bpm, ts, stale, error}.

    `stale` is True when the most recent reading is older than STALE_AFTER_S,
    so the caller can distinguish "live" from "frozen/disconnected".
    """
    snap = _latest.copy()
    ts = snap.get("ts")
    snap["stale"] = ts is None or (time.monotonic() - ts) > STALE_AFTER_S
    return snap


def is_running() -> bool:
    """True if the reader thread is alive."""
    return _thread is not None and _thread.is_alive()


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting watch HR reader (auto-connect). Ctrl+C to stop.")
    start()
    try:
        while True:
            time.sleep(1.0)
            r = get_hr()
            if r["error"]:
                print(f"  [error] {r['error']}")
            elif r["bpm"] is None:
                print("  connecting…")
            else:
                flag = " (STALE)" if r["stale"] else ""
                print(f"  ❤️ {r['bpm']} BPM{flag}")
    except KeyboardInterrupt:
        stop()
        print("\nStopped.")
