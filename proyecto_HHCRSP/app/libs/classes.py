from dataclasses import dataclass, field
from datetime import datetime
from dataclasses_json import dataclass_json
from typing import List, Dict, Any, Optional


@dataclass_json
@dataclass
class Nurse:
    nurse_id: str  # Unique identifier for the nurse
    weekend_preference: str  # (M) Morning or (A) afternoon
    accumulated_hours: int  # Number of accumlated hours from last month
    unavailability_days: List[str] = field(default_factory=list)  # List of days off
    days_off_requested: List[str] = field(default_factory=list)  # List of days off requested by the nurse (former DLI)
    available_morning_shifts: List[str] = field(default_factory=list)  # List of days available for morning shifts
    available_afternoon_shifts: List[str] = field(default_factory=list)  # List of days available for afternoon shifts


@dataclass_json
@dataclass
class Shift:
    shift_id:str  # Unique identifier for the shift
    shift_date: str# Date of the shift
    type_of_shift: str  # (M) Morning or (A) afternoon
    demand: int = 0  # Number of nurses needed for the shift
