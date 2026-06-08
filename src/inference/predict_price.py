from sklearn.exceptions import InconsistentVersionWarning
from functools import lru_cache
import numpy as np
import joblib
import warnings
import mlflow.pyfunc
import os

from src.registry.model_registry import (
    PRICE_MODEL_NAME,
    get_latest_model_version,
    get_model_path,
)

MODEL_NAME = PRICE_MODEL_NAME
MODEL_VERSION = get_latest_model_version(MODEL_NAME)
MODEL_PATH = get_model_path(MODEL_NAME, MODEL_VERSION)

@lru_cache(maxsize=1)
def _load_model():
    # model_path = MODEL_PATH
    # if not model_path.exists():
    #     detail = f"Pricing pipeline not found at {MODEL_PATH}."
    #     raise RuntimeError(detail)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)

        os.environ["AWS_PROFILE"] = "nicherides"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

        model_uri = (
            "s3://nicherides/mlflow-artifacts/1/models/"
            "m-b16472537faa4daf8ea86fe02d4d8d27/artifacts"
        )

        # model_name = "price-prediction-pipeline"

        # mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        # mlflow.set_tracking_uri(mlflow_uri)
        # mlflow.set_registry_uri(mlflow_uri)
        
        # model_uri = f"models:/{model_name}/latest"
        # model = mlflow.pyfunc.load_model(model_uri=model_uri)
        #model = joblib.load(model_path, mmap_mode="r")
        model = mlflow.sklearn.load_model(model_uri)

    return model

def predict(car_details):
    model = _load_model()
    predicted_price_log = model.predict(car_details)[0]

    predicted_price = np.expm1(predicted_price_log)

    if predicted_price <= 0:
        raise RuntimeError("Pricing model returned an invalid prediction.")
    
    return int(round(predicted_price)), MODEL_NAME, MODEL_VERSION
