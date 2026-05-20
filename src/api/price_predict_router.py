from fastapi import APIRouter
from datetime import datetime
import pandas as pd


from src.inference.predict_price import predict
from src.api.schemas import CarFeatures

router = APIRouter()

@router.post("/v1/price/predict")
def predict_price(data: CarFeatures):
    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    price, model_name, model_version = predict(pd.DataFrame([payload]))

    with open('../reports/price_model_report.md', 'a') as f:
        f.write(f"Time & Date: {datetime.now()} Predicted Price: {price}  \
                Model Name: {model_name} Model Version: {model_version}  \
                Make: {payload['make']} Model: {payload['model']}  \
                Year: {payload['year']} mileage: {payload['mileage']}")

    return {
            "model_name": model_name, 
            "model_version": model_version,
            "prediction": price
           }