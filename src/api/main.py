from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI

from src.api.vin_router import router as vin_router
from src.api.price_predict_router import router as price_predict_router

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="Car Price Predictor API")

app.include_router(vin_router)
app.include_router(price_predict_router)

@app.get("/health")
def health():
    return { "ok": True }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000)
