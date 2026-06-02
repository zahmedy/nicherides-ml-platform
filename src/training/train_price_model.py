from sklearn.model_selection import train_test_split
from pathlib import Path
from numpy import savetxt
import mlflow
import pandas as pd
import joblib
import datetime
import numpy as np
import os

from src.registry.model_registry import (
    PRICE_MODEL_NAME,
    get_model_path,
    get_next_model_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_model" / "raw" / "app_ready_.csv"
MODEL_NAME = PRICE_MODEL_NAME
MODEL_VERSION = get_next_model_version(MODEL_NAME)
MODEL_OUT_PATH = get_model_path(MODEL_NAME, MODEL_VERSION)
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
    training_model = model_pipeline["training"]

    # fix skewness
    y_train_log = np.log1p(y_train)

    model_pipeline.fit(X_train, y_train_log)

    y_pred_log = model_pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # basic metrics
    mae, rmse, r2, mape = get_basic_metrics(y_test, y_pred)

    # get cross validation
    mae_scores = get_cross_val_scores(model_pipeline, X_train, y_train)

    # get current mode metrics
    with open(MODEL_METRICS_REPORT, "a+") as f:
        f.seek(0)
        content = f.read()
        lines = content.splitlines()

        register = False

        if not lines:
            register =True
            metrcis = None
        else:
            metrcis = lines[-1]

            last_mae = metrcis.split()[4]
            last_rmse = metrcis.split()[6]
            last_r2 = metrcis.split()[8]
            last_mape = metrcis.split()[10]
            last_cv_mae = metrcis.split()[-1]

            # new metrics is better save model 
            if float(last_mae.replace('$', '').replace(',', '')) > mae \
                and float(last_mape.replace('%', '')) > mape \
                or float(last_r2) < r2:
                register = True


         # register with mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_registry_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("price-prediction")
        with mlflow.start_run():
            mlflow.log_param("num_trees", training_model.n_estimators)
            mlflow.log_param("maxdepth", training_model.max_depth)
            mlflow.log_param("min_sample_leaf", training_model.min_samples_leaf)

            savetxt('predictions.csv', y_pred, delimiter=',')
            mlflow.log_artifact('predictions.csv')

            # log to MLFLOW
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)
            mlflow.log_metric("CV mae", mae_scores.mean())

            os.remove('predictions.csv')

            if register:
                mlflow.sklearn.log_model(
                    sk_model=model_pipeline,
                    name="ExtraTree-price-model",
                    input_example=X_test,
                    registered_model_name='price-prediction-pipeline')

                f.write(
                f"DATE: {datetime.datetime.now()} "
                f"MAE: ${mae:,.0f} "
                f"RMSE: ${rmse:,.0f} "
                f"R^2: {r2:.3f} "
                f"MAPE: {mape:.2f}% "
                f"CV MAE: ${mae_scores.mean():,.0f}\n")

                # save model pipeline
                MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model_pipeline, MODEL_OUT_PATH)
            else:
                print(f"Training finished: ignorning as metrics are worse than or equal last run.")
                print(f"Old metrcis: {metrcis}")
                print(f"New metrics: "
                                    f"DATE: {datetime.datetime.now()} "
                                    f"MAE: ${mae:,.0f} "
                                    f"RMSE: ${rmse:,.0f} "
                                    f"R^2: {r2:.3f} "
                                    f"MAPE: {mape:.2f}% "
                                    f"CV MAE: ${mae_scores.mean():,.0f}\n")


if __name__ == "__main__":
    train()
