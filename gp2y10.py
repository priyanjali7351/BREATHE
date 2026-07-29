"""
gp2y10.py — Sharp GP2Y1010AU0F dust-sensor serial reader for BREATHE / BioAQI

Reads approximate PM2.5 (µg/m³) sent by the Arduino sketch
(arduino/gp2y10_reader) over USB serial and maps it to an India CPCB AQI.
Mirrors sensor.py's design: a background daemon thread keeps a shared latest
reading, and the app polls get_reading() without blocking.

This is the "Sensor mode" local-air source — the sensor measures the air
immediately around the user, complementing realtime.py's city-level API data
("Auto mode"). It reacts live: a match or incense near the sensor spikes PM2.5.

Honesty note: the GP2Y10 is a low-cost optical dust sensor, not a calibrated
reference-grade PM2.5 monitor. It detects coarse dust + smoke and is reported
as an *approximate* PM2.5 / relative particulate level.

Usage:
    import gp2y10
    gp2y10.start("COM3")
    r = gp2y10.get_reading()
    # {"pm25": 41.2, "aqi": 76.0, "raw": 183, "error": None}
"""

import threading

import serial
import serial.tools.list_ports

# Shared state — written by background thread, read by Streamlit/the app.
_latest: dict = {"pm25": None, "aqi": None, "raw": None, "error": None}
_thread: threading.Thread | None = None


# ── PM2.5 → India CPCB AQI (matches realtime._pm25_to_aqi) ─────────────────────

def pm25_to_aqi(pm25: float) -> float:
    """India CPCB 24-hr PM2.5 sub-index (piecewise linear). Kept identical to
    realtime.py so the sensor path and the API path produce comparable AQIs."""
    breakpoints = [
        (0.0,   30.0,   0,   50),
        (30.0,  60.0,  51,  100),
        (60.0,  90.0, 101,  200),
        (90.0, 120.0, 201,  300),
        (120.0, 250.0, 301, 400),
        (250.0, 500.0, 401, 500),
    ]
    pm25 = max(0.0, min(float(pm25 or 0.0), 500.0))
    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if pm25 <= c_hi:
            return round(aqi_lo + (pm25 - c_lo) / (c_hi - c_lo) * (aqi_hi - aqi_lo), 1)
    return 500.0


# ── Line parsing (pure function — unit-testable without hardware) ─────────────

def parse_line(line: str) -> dict | None:
    """Parse one serial line like 'PM25:41.2,RAW:183' → reading dict, or None.

    Returns None for non-data lines (INFO:..., blank, malformed) so the caller
    can simply skip them.
    """
    line = line.strip()
    if not line.startswith("PM25:"):
        return None
    try:
        parts = line.split(",")
        pm25 = float(parts[0].split(":", 1)[1])
        raw = int(parts[1].split(":", 1)[1]) if len(parts) > 1 else None
        return {
            "pm25": round(pm25, 1),
            "aqi": pm25_to_aqi(pm25),
            "raw": raw,
            "error": None,
        }
    except (ValueError, IndexError):
        return None  # malformed line — ignore


# ── Background serial reader ───────────────────────────────────────────────────

def _read_loop(port: str, baud: int = 9600) -> None:
    """Runs in a daemon thread. Parses 'PM25:<ugm3>,RAW:<adc>' lines."""
    global _latest
    try:
        ser = serial.Serial(port, baud, timeout=3)
        while True:
            raw_line = ser.readline().decode("utf-8", errors="ignore")
            reading = parse_line(raw_line)
            if reading is not None:
                _latest = reading
    except serial.SerialException as exc:
        _latest = {"pm25": None, "aqi": None, "raw": None, "error": str(exc)}
    except Exception as exc:
        _latest = {"pm25": None, "aqi": None, "raw": None, "error": f"Unexpected: {exc}"}


# ── Public API (mirrors sensor.py) ─────────────────────────────────────────────

def start(port: str, baud: int = 9600) -> None:
    """Start the background serial reader thread (call once on connect)."""
    global _thread, _latest
    _latest = {"pm25": None, "aqi": None, "raw": None, "error": None}
    _thread = threading.Thread(
        target=_read_loop,
        args=(port, baud),
        daemon=True,   # exits automatically when the app process exits
        name="gp2y10-reader",
    )
    _thread.start()


def get_reading() -> dict:
    """Latest sensor reading (thread-safe snapshot).
    Keys: pm25 (float|None), aqi (float|None), raw (int|None), error (str|None)."""
    return _latest.copy()


def list_ports() -> list[str]:
    """Return all available serial port names (e.g. ['COM3', 'COM4'])."""
    return [p.device for p in serial.tools.list_ports.comports()]


def is_running() -> bool:
    """True if the reader thread is alive."""
    return _thread is not None and _thread.is_alive()
