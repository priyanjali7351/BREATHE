"""api/schemas.py — request/response models for the BREATHE FastAPI backend."""

from pydantic import BaseModel, Field


class SensorReadingIn(BaseModel):
    device_id: str
    pm25: float
    pm10: float
    temp_c: float
    humidity: float
    lat: float | None = None   # only needed the first time a device reports
    lon: float | None = None
    is_real: bool = True       # False for simulated IDW-demo nodes


class VitalsIn(BaseModel):
    user_id: str
    hr: float
    spo2: float


class PredictIn(BaseModel):
    user_id: str
    lat: float
    lon: float
    age: int = 30
    conditions: list[str] = Field(default_factory=lambda: ["Healthy"])
    activity_level: str = "Sedentary"
    hours_outdoors: float = 1.0
    trust_radius_m: float = 500.0   # IDW trust radius; doc default is 500m


class PredictOut(BaseModel):
    phrs: float
    category: str
    color: str
    aqi_used: float
    source: str          # "idw_mesh" | "nearest_fallback_out_of_radius" | "no_data"
    device_id: str | None = None
    n_sensors_used: int | None = None