
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
    df_filtered_columns = filter_columns(df)
    df_filter_bad_rows = filter_bad_rows(df_filtered_columns)
    df_drop_more_than_4_nans = drop_more_than_4_nans(df_filter_bad_rows)
    X, y = train_test_split(df_drop_more_than_4_nans, test_size=0.2, random_state=42)
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

    def testMakeCanonicalizerOther(self):
        canonicalizer = MakeCanonicalizer()
        # we have less than 30 BMW so it should be turned to other 
        make_count = (self.df['make'] == 'BMW').sum()
        self.assertTrue(make_count < 30)
        canonicalizer.fit(self.X)
        X = canonicalizer.transform(self.X)
        self.assertTrue(X['make'].str.lower().str.contains('other').all())
    
    def testModelCanonicalizerOther(self):
        canonicalizer = ModelCanonicalizer()
        # we have less than 10 Panamera so it should be turned to other 
        model_count = (self.df['model'] == 'Panamera').sum()
        self.assertTrue(model_count < 10)
        canonicalizer.fit(self.X)
        X = canonicalizer.transform(self.X)
        self.assertTrue(X['model'].str.lower().str.contains('other').all())


if __name__ == "__main__":
    unittest.main()