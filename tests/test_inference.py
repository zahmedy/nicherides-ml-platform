import unittest
import pandas as pd
import numpy as np
import mlflow

from src.inference.predict_price import MODEL_PATH, predict

class TestMlflowModelLoad(unittest.TestCase):
    def test_loading_model_from_mlflow_rigestry(self):
        model_name = "price-prediction-pipeline"

        # Load a specific version
        model_uri = f"models:/{model_name}/1"
        mlflow.set_registry_uri("http://127.0.0.1:5000")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        car_details = pd.DataFrame([{
            "make": "Toyota",
            "model": "Camry",
            "year": 2018.0,
            "body_type": "SUV",
            "fuel_type": "Petrol",
            "engine_volume": 2.0,
            "mileage": 46073.0,
            "engine_cylinders": 4.0,
            "transmission": "Automatic",
            "drivetrain": "FWD",
            "color": "White"
        }])
        predicted_price_log = model.predict(car_details)[0]
        predicted_price = np.expm1(predicted_price_log)
        print(predicted_price)
        self.assertTrue(500 <= predicted_price <= 500000)

class TestTraining(unittest.TestCase):
    def test_training_pipeline(self):
        self.assertTrue(MODEL_PATH.exists())

class TestInference(unittest.TestCase):
    def test_inference_load_file(self):
        car_details = pd.DataFrame([{
            "make": "Toyota",
            "model": "Camry",
            "year": 2018.0,
            "body_type": "SUV",
            "fuel_type": "Petrol",
            "engine_volume": 2.0,
            "mileage": 46073.0,
            "engine_cylinders": 4.0,
            "transmission": "Automatic",
            "drivetrain": "FWD",
            "color": "White"
        }])
        predicted_price = predict(car_details)[0]
        print(predicted_price)
        self.assertTrue(500 <= predicted_price <= 500000)


if __name__ == "__main__":
    unittest.main()
