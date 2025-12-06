from typing import Dict,Any
from sqlmodel import SQLModel, Field,Column
from sqlalchemy.dialects.postgresql import JSONB
import uuid

class Predictor(SQLModel, table=True):
    __tablename__='predictor'
    id:uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    patient_id: str
    model_name: str
    params: Dict[str, Any] = Field(sa_column=Column(JSONB))
    accuracy:float
    prediction :str





