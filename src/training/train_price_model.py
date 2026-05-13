from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_model" / "processed" / "app_ready.csv"

from src.pipelines.car_prices_pipelines import get_model_pipeline
from src.evaluation.evaluate_regression import get_mae_and_r2

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

    mae, r2 = get_mae_and_r2(y_test, y_pred)

    print(f"MAE: {mae}.  R^2: {r2}")



if __name__ == "__main__":
    train()
