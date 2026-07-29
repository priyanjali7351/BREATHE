"""
api/aqi_convert.py — PM2.5 -> India CPCB AQI sub-index conversion.

Your sensor node reports raw PM2.5/PM10 concentrations, not an AQI value.
This did not exist anywhere in the repo, so it's added here rather than
silently assumed. Breakpoints are CPCB's published 24-hr PM2.5 sub-index
table (National Air Quality Index, CPCB 2014) — the same standard your
disclosure doc anchors on elsewhere (normalize_aqi_india in preprocess.py).
"""

import numpy as np

# (C_low, C_high, I_low, I_high) — concentration in µg/m3, index 0-500
_PM25_BREAKPOINTS = [
    (0,   30,    0,  50),
    (31,  60,   51, 100),
    (61,  90,  101, 200),
    (91,  120, 201, 300),
    (121, 250, 301, 400),
    (251, 350, 401, 500),
    (351, 500, 501, 500),  # saturate beyond table
]


def pm25_to_aqi(pm25: float) -> float:
    """CPCB piecewise-linear sub-index formula: I = ((I_hi-I_lo)/(C_hi-C_lo)) * (C-C_lo) + I_lo"""
    pm25 = max(0.0, pm25)
    for c_lo, c_hi, i_lo, i_hi in _PM25_BREAKPOINTS:
        if pm25 <= c_hi:
            return round(i_lo + (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo), 1)
    return 500.0