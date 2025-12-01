from pydantic import BaseModel,Field,field_validator
from enum import Enum 
class PredictionResponse(BaseModel):
    prediction: str  
    probability_class_yes: float
    probability_class_no: float
class Gender(str,Enum):
    M='M'
    F='F'

class Binary(int,Enum):
    ZERO = 0
    ONE = 1

class Three_Classes(int,Enum):
    ZERO = 0
    ONE = 1
    TWO = 2

class Four_Classes(int,Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    
    
class Seizures(str,Enum):
    M='M'
    TC="TC"
    S="S"
    C='C'
    

class input_ml  (BaseModel):
    age: int =Field()
    gender: Gender =Field()
    antecedents: Four_Classes
    headaches:Three_Classes
    seizures:Seizures 
    fatigue: Three_Classes
    drowsiness: Three_Classes
    sleep_pb: Three_Classes
    memory_pb: Three_Classes
    pressure: Four_Classes
    balance_loss: Binary
    judgment_degradation: Four_Classes
    sense_degradation: Four_Classes
    lactation: Three_Classes
    swallowing: Four_Classes
    muscle: Four_Classes
    @field_validator("age")
    def validate_age(cls,age):
        if age<=0 or age >130:
            raise ValueError("Invalid age value")
        return age 


