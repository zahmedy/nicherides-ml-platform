import unittest
from pathlib import Path
import pandas as pd

from src.training.train_price_model import train
from src.inference.predict_price import predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "price_model" / "v1" / "car_price_pipeline.pkl"


class TestTraining(unittest.TestCase):
    def test_training_pipeline(self):
        train()
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
        predicted_price = predict(car_details)
        print(predicted_price)
        self.assertTrue(5000 <= predicted_price <= 20000)


if __name__ == "__main__":
    unittest.main()