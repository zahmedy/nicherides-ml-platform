
import unittest
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from src.features.build_features import (ModelTargetEncoder, 
                                         MakeCanonicalizer, 
                                         ModelCanonicalizer,
                                         FeatureEngineering)
from src.data.clean_data import filter_columns, filter_bad_rows, drop_more_than_4_nans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_model" / "raw" / "cars_listing_testing_set.csv"

df = pd.read_csv(RAW_DATA_PATH)
df_filtered_columns = filter_columns(df)
df_filter_bad_rows = filter_bad_rows(df_filtered_columns)
df_drop_more_than_4_nans = drop_more_than_4_nans(df_filter_bad_rows)
train, test = train_test_split(df_drop_more_than_4_nans, test_size=0.2, random_state=42)

X = train.drop(columns=['price'], axis=1)
y = train['price']


class TestMakeCanonicalizer(unittest.TestCase):
    def test_make_canonicalizer(self):
        X = pd.DataFrame({
            "make": ["ToyoTa", "Toyota!", "KiA", "Unk", "FORD!!!"],
            "year": [2002, 2022, 2010, 2019, 2000]
        })

        y = pd.Series([12000, 25000, 12000, 25000, 12000])

        # prepare make_canonicalizer
        make_canonicalizer = MakeCanonicalizer(min_count=1)
        make_canonicalizer_other = MakeCanonicalizer(min_count=2)
        make_canonicalizer.fit(X, y)
        make_canonicalizer_other.fit(X, y)
        transformed = make_canonicalizer.transform(X)
        transformed_other = make_canonicalizer_other.transform(X)

        self.assertEqual(transformed['make'].tolist(), ["Toyota", "Toyota", "Kia", "other", "Ford"])
        self.assertEqual(transformed_other['make'].tolist(), ["Toyota", "Toyota", "other", "other", "other"] )

class TestModelCanonicalizer(unittest.TestCase):
    def test_model_canonicalizer(self):
        X = pd.DataFrame({
            "model": ["camry!", "ES350", "rav 4", "FakeModel", "priusc", "camRY"],
            "year": [2002, 2022, 2010, 2019, 2000, 2022]
        })

        y = pd.Series([12000, 25000, 12000, 25000, 12000, 44000])

        # prepare make_canonicalizer
        model_canonicalizer = ModelCanonicalizer(min_count=1)
        model_canonicalizer_other = ModelCanonicalizer(min_count=2)
        model_canonicalizer.fit(X, y)
        model_canonicalizer_other.fit(X, y)
        transformed = model_canonicalizer.transform(X)
        transformed_other = model_canonicalizer_other.transform(X)

        self.assertEqual(transformed['model'].tolist(), ["Camry", "ES 350", "RAV4", "other", "Prius C", "Camry"])
        self.assertEqual(transformed_other['model'].tolist(), ["Camry", "other", "other", "other", "other", "Camry"])

class TestFeatureEngineering(unittest.TestCase):
    def test_feature_engineering(self):
        # prepare feature engineering
        feature_engin = FeatureEngineering()
        feature_engin.fit(X, y)
        feature_engin_X = feature_engin.transform(X)

        created_features = ["car_age", "engine_displacement", "miles_per_year"]
        for feature in created_features:
            self.assertTrue(feature in feature_engin_X.columns)

class TestModelTargetEncoder(unittest.TestCase):
    def test_model_target_encoder(self):
        X = pd.DataFrame({
            "model": ["Camry", "Camry", "RAV4", "RAV4", "Prius"],
            "year": [2020, 2021, 2020, 2021, 2019],
        })
        y = pd.Series([20000, 22000, 30000, 34000, 18000])

        encoder = ModelTargetEncoder(alpha=1)
        encoder.fit(X, y)

        transformed = encoder.transform(pd.DataFrame({
            "model": ["Camry", "RAV4", "Prius", "Unknown"],
            "year": [2022, 2022, 2022, 2022]
        }))

        self.assertIn('model_te', transformed.columns)
        self.assertNotIn('model', transformed.columns)

        # Global mean = 24800
        global_mean = y.mean()
        self.assertAlmostEqual(transformed.loc[0, "model_te"], 22266.667, places=2)
        self.assertAlmostEqual(transformed.loc[1, "model_te"], 29600.0, places=2)
        self.assertAlmostEqual(transformed.loc[2, "model_te"], 21400.0, places=2)
        self.assertAlmostEqual(transformed.loc[3, "model_te"], global_mean, places=2)


if __name__ == "__main__":
    unittest.main()