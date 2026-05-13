
import unittest
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from src.features.build_features import (ModelTargetEncoder, 
                                         MakeCanonicalizer, 
                                         ModelCanonicalizer, 
                                         SUPPORTED_MAKES, SUPPORTED_MODELS)
from src.data.clean_data import filter_columns, filter_bad_rows, drop_more_than_4_nans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_model" / "raw" / "cars_listing_testing_set.csv"


class TestFeatures(unittest.TestCase):

    df = pd.read_csv(RAW_DATA_PATH)
    df = filter_columns(df)
    df = filter_bad_rows(df)
    df = drop_more_than_4_nans(df)
    X, y = train_test_split(df, test_size=0.2, random_state=42)
    X_train = X.drop('price', axis=1)
    X_test = X['price']
    y_train = y.drop('price', axis=1)
    y_test = y['price'] 

    def testMakeCanonicalizer(self):
        canonicalizer = MakeCanonicalizer()
        canonicalizer.fit(self.X)
        X = canonicalizer.transform(self.X)
        supported_makes = [make.lower() for make in SUPPORTED_MAKES]
        X = X.query("make != 'other'")
        self.assertTrue(X['make'].str.lower().isin(supported_makes).all())

    def testModelCanonicalizer(self):
        canonicalizer = ModelCanonicalizer()
        canonicalizer.fit(self.X, self.y)
        X = canonicalizer.transform(self.X)
        supported_models = [model.lower() for model in SUPPORTED_MODELS]
        X = X.query("model != 'other'")
        self.assertTrue(X['model'].str.lower().isin(supported_models).all())

if __name__ == "__main__":
    unittest.main()