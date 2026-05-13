import unittest
from pathlib import Path

import pandas as pd
from src.data.clean_data import filter_columns, filter_bad_rows, drop_more_than_4_nans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_model" / "raw" / "cars_listing.csv"


class TestFeatures(unittest.TestCase):

    df = pd.read_csv(RAW_DATA_PATH)

    def test_filter_columns(self):
        filtered_df = filter_columns(self.df)
        self.assertEqual(len(filtered_df.columns), 12)

    def test_bad_rows(self):
        filtered_df = filter_bad_rows(self.df)
        self.assertTrue(filtered_df[filtered_df['price'] < 0].empty)
        self.assertTrue(filtered_df[filtered_df['price'] > 500000].empty)
        self.assertTrue(filtered_df[filtered_df['year'] > 2027].empty)
        self.assertTrue(filtered_df[filtered_df['year'] < 1970].empty)
        self.assertTrue(filtered_df[filtered_df['mileage'] > 500000].empty)
        self.assertTrue(filtered_df[filtered_df['mileage'] < 2].empty)
        self.assertTrue(filtered_df[filtered_df['engine_volume'] > 7].empty)
        self.assertTrue(filtered_df[filtered_df['engine_volume'] < 1].empty)
        self.assertTrue(filtered_df[filtered_df['engine_cylinders'] > 16].empty)
        self.assertTrue(filtered_df[filtered_df['engine_cylinders'] < 3].empty)

    def test_no_more_than_3_nan(self):
        filtered_df = drop_more_than_4_nans(self.df)
        self.assertTrue(filtered_df[filtered_df.isna().sum(axis=1) > 3].empty)

if __name__ == "__main__":
    unittest.main()