"""
api/idw.py — Inverse Distance Weighting spatial interpolation.

Per your disclosure doc (Section 7, Claim 4): power p=2, default trust
radius 500 m, falls back when no sensor node lies within the radius.

Fallback behavior here (Step 2 scope): if no sensor is within trust_radius_m,
returns None and the caller decides what to do — currently falls back to the
single nearest sensor regardless of distance, clearly flagged in the response
`source` field. A real city-snapshot fallback (per the doc) isn't built yet;
that's a later addition once you have city-level data, not invented here.
"""

import math
from dataclasses import dataclass


@dataclass
class SensorPoint:
    device_id: str
    lat: float
    lon: float
    pm25: float
    pm10: float


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def idw_estimate(
    lat: float,
    lon: float,
    sensors: list[SensorPoint],
    power: float = 2.0,
    trust_radius_m: float = 500.0,
) -> dict | None:
    """
    Returns {"pm25": .., "pm10": .., "n_sensors_used": .., "distances_m": [..]}
    or None if no sensor lies within trust_radius_m (caller must handle fallback).
    """
    in_range = []
    for s in sensors:
        d = _haversine_m(lat, lon, s.lat, s.lon)
        if d <= trust_radius_m:
            in_range.append((s, d))

    if not in_range:
        return None

    # If the query point sits essentially on top of a sensor, avoid div-by-zero
    # and just return that sensor's exact reading.
    for s, d in in_range:
        if d < 1.0:
            return {
                "pm25": s.pm25, "pm10": s.pm10,
                "n_sensors_used": 1, "distances_m": [round(d, 1)],
            }

    weights = [1.0 / (d ** power) for _, d in in_range]
    total_w = sum(weights)

    pm25 = sum(w * s.pm25 for (s, _), w in zip(in_range, weights)) / total_w
    pm10 = sum(w * s.pm10 for (s, _), w in zip(in_range, weights)) / total_w

    return {
        "pm25": round(pm25, 1),
        "pm10": round(pm10, 1),
        "n_sensors_used": len(in_range),
        "distances_m": [round(d, 1) for _, d in in_range],
    }