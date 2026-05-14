from sklearn.exceptions import InconsistentVersionWarning
from pathlib import Path
from functools import lru_cache
import numpy as np
import joblib
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "price_model" / "v1" / "car_price_pipeline.pkl"

@lru_cache(maxsize=1)
def _load_model():
    model_path = MODEL_PATH
    if not model_path.exists():
        detail = f"Pricing pipeline not found at {MODEL_PATH}."
        raise RuntimeError(detail)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        model = joblib.load(model_path, mmap_mode="r")

    return model

def predict(car_details):
    model = _load_model()
    predicted_price_log = model.predict(car_details)[0]

    predicted_price = np.expm1(predicted_price_log)

    if predicted_price <= 0:
        raise RuntimeError("Pricing model returned an invalid prediction.")
    
    return int(round(predicted_price))
