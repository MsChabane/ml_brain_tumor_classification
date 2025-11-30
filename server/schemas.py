from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str  
    probability_class_yes: float
    probability_class_no: float
