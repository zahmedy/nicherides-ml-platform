import unittest
from pathlib import Path

import pandas as pd
from src.data.clean_data import drop_more_than_4_nans
from src.data.validate_data import filter_columns, filter_bad_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_model" / "raw" / "cars_listing_testing_set.csv"


class TestDataCleaning(unittest.TestCase):
    def test_drop_nans(self):
        df = pd.read_csv(RAW_DATA_PATH)
        transformed = drop_more_than_4_nans(df)

        self.assertTrue(transformed[transformed.isna().sum(axis=1) > 3].empty)

    
class TestDataFiltering(unittest.TestCase):
    df = pd.read_csv(RAW_DATA_PATH)

    def test_filter_columns(self):
        filtered_df = filter_columns(self.df)
        self.assertEqual(filtered_df.columns.tolist(), [
                "price", "make", "model", "year", "body_type",
                "fuel_type", "engine_volume", "mileage",
                "engine_cylinders", "transmission", "drivetrain", "color",
            ])

    
    def test_bad_rows(self):
        df = pd.read_csv(RAW_DATA_PATH)
        df = filter_bad_rows(df)

        self.assertTrue(df[df['price'] < 0].empty)
        self.assertTrue(df[df['price'] > 500000].empty)
        self.assertTrue(df[df['year'] > 2027].empty)
        self.assertTrue(df[df['year'] < 1970].empty)
        self.assertTrue(df[df['mileage'] > 500000].empty)
        self.assertTrue(df[df['mileage'] < 2].empty)
        self.assertTrue(df[df['engine_volume'] > 7].empty)
        self.assertTrue(df[df['engine_volume'] < 1].empty)
        self.assertTrue(df[df['engine_cylinders'] > 16].empty)
        self.assertTrue(df[df['engine_cylinders'] < 3].empty)

if __name__ == "__main__":
    unittest.main()