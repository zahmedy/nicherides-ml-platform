from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict_price import predict
from src.api.schemas import CarFeatures

app = FastAPI(title="Car Price Predictor API")

@app.post("/predict")
def predict_price(data: CarFeatures):
    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    price = predict(pd.DataFrame([payload]))

    return { "price": price }

@app.get("/health")
def health():
    return { "ok": True }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000)
