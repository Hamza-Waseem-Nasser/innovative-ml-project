# api/schemas.py
from pydantic import BaseModel
from typing import List

class TransactionFeatures(BaseModel):
    StockCode: str
    Country: str
    Quantity: int
    Price: float
    hour: int
    day_of_week: int
    month: int

class PredictionResponse(BaseModel):
    TotalPrice: float
