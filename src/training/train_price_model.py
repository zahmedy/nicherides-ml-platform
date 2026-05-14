from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import joblib
import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_model" / "processed" / "app_ready.csv"
MODEL_OUT_PATH = PROJECT_ROOT / "models" / "price_model" / "v1" / "car_price_pipeline.pkl"
MODEL_METRICS_REPORT = PROJECT_ROOT / "reports" / "price_model_report.md"

from src.pipelines.car_prices_pipelines import get_data_quality_pipeline, get_model_pipeline
from src.evaluation.evaluate_regression import get_basic_metrics, get_cross_val_scores

def train():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    data_quality_pipeline = get_data_quality_pipeline()
    df = data_quality_pipeline.fit_transform(df)

    # Split to train and test so no data leakage occure when target encoding the car model
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train.drop("price", axis=1)
    y_train = train["price"]

    X_test = test.drop("price", axis=1)
    y_test = test["price"]

    model_pipeline = get_model_pipeline()

    # fix skewness
    y_train_log = np.log1p(y_train)

    model_pipeline.fit(X_train, y_train_log)

    y_pred_log = model_pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # basic metrics
    mae, rmse, r2, mape = get_basic_metrics(y_test, y_pred)

    # get cross validation
    mae_scores = get_cross_val_scores(model_pipeline, X_train, y_train)
    
    with open(MODEL_METRICS_REPORT, "a") as f:
        f.write(
        f"DATE: {datetime.datetime.now()} "
        f"MAE: ${mae:,.0f} "
        f"RMSE: ${rmse:,.0f} "
        f"R^2: {r2:.3f} "
        f"MAPE: {mape:.2f}% "
        f"CV MAE: ${mae_scores.mean():,.0f}\n")

    # save model pipeline
    joblib.dump(model_pipeline, MODEL_OUT_PATH)


if __name__ == "__main__":
    train()
