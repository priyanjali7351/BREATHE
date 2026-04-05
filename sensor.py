"""
sensor.py — MQ135 serial reader for BREATHE / BioAQI

Reads values sent by the Arduino sketch over USB serial and converts
them to an approximate India CPCB AQI value using the RAW ADC reading.

Note on PPM: The datasheet power-law formula (PPM = A * (Rs/Ro)^B) requires
a proper per-unit lab calibration to give absolute CO₂ values. Without that,
PPM values will be in the 1–20 range for typical indoor air. We therefore
derive AQI from the raw ADC value, which is always reliable as a relative
air-quality indicator.

RAW ADC reference (MQ135 on 5V Arduino):
  ~100–200  →  Very clean / outdoor air  (AQI Good)
  ~200–350  →  Normal indoor air         (AQI Satisfactory–Moderate)
  ~350–500  →  Mild indoor pollution     (AQI Moderate–Poor)
  ~500–700  →  Elevated VOC / light smoke (AQI Poor–Very Poor)
  ~700+     →  Heavy smoke               (AQI Severe)

Usage:
    import sensor
    sensor.start("COM3")          # starts background thread
    reading = sensor.get_reading()
    # {"ppm": 5.2, "aqi": 120, "raw": 362, "error": None}
"""

import threading
import serial
import serial.tools.list_ports

# Shared state — written by background thread, read by Streamlit
_latest: dict = {"ppm": None, "aqi": None, "raw": None, "error": None}
_thread: threading.Thread | None = None


# ── RAW ADC → AQI conversion ──────────────────────────────────────────────────
# Breakpoints derived from typical MQ135 behaviour on a 5V Arduino.
# (raw_threshold, aqi_value)
_RAW_BREAKPOINTS = [
    (100,   0),   # very clean outdoor air
    (200,  50),   # clean indoor air
    (350, 150),   # normal indoor / light VOCs
    (500, 250),   # mild pollution
    (650, 350),   # elevated smoke / fumes
    (850, 450),   # heavy smoke
    (1023, 500),  # sensor saturation
]


def _raw_to_aqi(raw: int) -> int:
    """Linearly interpolate RAW ADC to AQI using empirical breakpoints."""
    if raw <= _RAW_BREAKPOINTS[0][0]:
        return 0
    for i in range(1, len(_RAW_BREAKPOINTS)):
        r_lo, a_lo = _RAW_BREAKPOINTS[i - 1]
        r_hi, a_hi = _RAW_BREAKPOINTS[i]
        if raw <= r_hi:
            ratio = (raw - r_lo) / (r_hi - r_lo)
            return int(a_lo + ratio * (a_hi - a_lo))
    return 500


# ── Background serial reader ──────────────────────────────────────────────────

def _read_loop(port: str, baud: int = 9600) -> None:
    """Runs in a daemon thread. Parses lines like 'PPM:5.2,RAW:362'."""
    global _latest
    try:
        ser = serial.Serial(port, baud, timeout=3)
        while True:
            line_bytes = ser.readline()
            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if line.startswith("PPM:"):
                try:
                    # Format: "PPM:5.2,RAW:362" (or legacy "PPM:5.2")
                    parts = line.split(",")
                    ppm = float(parts[0].split(":", 1)[1])
                    adc = int(parts[1].split(":", 1)[1]) if len(parts) > 1 else None
                    aqi = _raw_to_aqi(adc) if adc is not None else 0
                    _latest = {
                        "ppm": round(ppm, 1),
                        "aqi": aqi,
                        "raw": adc,
                        "error": None,
                    }
                except (ValueError, IndexError):
                    pass  # malformed line — ignore
    except serial.SerialException as exc:
        _latest = {"ppm": None, "aqi": None, "raw": None, "error": str(exc)}
    except Exception as exc:
        _latest = {"ppm": None, "aqi": None, "raw": None, "error": f"Unexpected: {exc}"}


# ── Public API ────────────────────────────────────────────────────────────────

def start(port: str, baud: int = 9600) -> None:
    """Start the background serial reader thread (call once on connect)."""
    global _thread, _latest
    # Reset state before starting
    _latest = {"ppm": None, "aqi": None, "raw": None, "error": None}
    _thread = threading.Thread(
        target=_read_loop,
        args=(port, baud),
        daemon=True,   # exits automatically when Streamlit process exits
        name="mq135-reader",
    )
    _thread.start()


def get_reading() -> dict:
    """
    Returns the latest sensor reading (thread-safe snapshot).
    Keys: ppm (float|None), aqi (int|None), error (str|None)
    """
    return _latest.copy()


def list_ports() -> list[str]:
    """Return all available serial port names (e.g. ['COM3', 'COM4'])."""
    return [p.device for p in serial.tools.list_ports.comports()]


def is_running() -> bool:
    """True if the reader thread is alive."""
    return _thread is not None and _thread.is_alive()
