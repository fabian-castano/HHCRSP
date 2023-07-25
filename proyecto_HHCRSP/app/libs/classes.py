from dataclasses import dataclass, field
from datetime import datetime
from dataclasses_json import dataclass_json
from typing import List, Dict, Any, Optional


@dataclass_json
@dataclass
class Nurse:
    nurse_id: int # Unique identifier for the nurse
    nurse_name: str  # Name of the nurse

    shift_preference: str  # (M) Morning or (A) afternoon
    accumulated_hours: float  # Number of accumlated hours from last month
    
    morning_availability_labor_day: bool
    morning_availability_weekend: bool
    afternoon_availability_labor_day: bool
    afternoon_availability_weekend: bool

    dates_off: List[int] = field(default_factory=list)  # List of dates off for the nurse
    vacations: List[int] = field(default_factory=list)  # List of vacations for the nurse

"""'ID',
 'name',
   'vacations',
     'dates_off',
       'TF',
       'morning_availability_labor_day',
         'morning_availability_weekend',
       'afternoon_availability_labor_day',
         'afternoon_availability_weekend',
       'DLI'"""

@dataclass_json
@dataclass
class Shift:
    shift_id:str  # Unique identifier for the shift
    shift_date: str# Date of the shift
    shift: str  # (M, COM, DISP,  T1, T2, CHX)
    shift_type: str  # (M) Morning or (A) afternoon
    weekday: int 
    week_of_year: int
    demand: int = 0  # Number of nurses needed for the shift
    holiday: bool = False  # True if the shift is a holiday
    

