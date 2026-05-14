from pydantic import BaseModel
from typing import Optional

class CarFeatures(BaseModel):
    # city: Optional[str] = None
    # district: Optional[str] = None
    make: str
    model: str
    year: int
    mileage: Optional[int] = None
    body_type: Optional[str] = None
    transmission: Optional[str] = None
    fuel_type: Optional[str] = None
    drivetrain: Optional[str] = None
    engine_cylinders: Optional[int] = None
    engine_volume: Optional[float] = None
    condition: Optional[str] = None
    color: Optional[str] = None
    # title: Optional[str] = None
    # description: Optional[str] = None