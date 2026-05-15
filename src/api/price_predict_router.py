from fastapi import APIRouter
import pandas as pd

from src.inference.predict_price import predict
from src.api.schemas import CarFeatures

router = APIRouter()

@router.post("/v1/price/predict")
def predict_price(data: CarFeatures):
    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    price = predict(pd.DataFrame([payload]))

    return { "price": price }