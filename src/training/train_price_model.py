from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import joblib
import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_model" / "processed" / "app_ready.csv"
MODEL_OUT_PATH = PROJECT_ROOT / "models" / "price_model" / "v1" / "car_price_pipeline.pkl"
MODEL_METRICS_REPORT = PROJECT_ROOT / "reports" / "price_model_report.md"

from src.pipelines.car_prices_pipelines import get_model_pipeline
from src.evaluation.evaluate_regression import get_basic_metrics, get_cross_val_scores

def train():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Split to train and test so no data leakage occure when target encoding the car model
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train.drop("price", axis=1)
    y_train = train["price"]

    X_test = test.drop("price", axis=1)
    y_test = test["price"]

    model_pipeline = get_model_pipeline()
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

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
