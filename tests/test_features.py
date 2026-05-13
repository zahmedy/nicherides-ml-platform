
import unittest
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from src.features.build_features import (ModelTargetEncoder, 
                                         MakeCanonicalizer, 
                                         ModelCanonicalizer,
                                         FeatureEngineering, 
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

    # prepare make_canonicalizer
    make_canonicalizer = MakeCanonicalizer()
    make_canonicalizer.fit(X, y)
    X_make_canonicalized = make_canonicalizer.transform(X)

    # prepare model_canonicalizer
    model_canonicalizer = ModelCanonicalizer()
    model_canonicalizer.fit(X, y)
    X_model_canonicalized = model_canonicalizer.transform(X_make_canonicalized)
   
    # prepare feature engineering
    feature_engin = FeatureEngineering()
    feature_engin.fit(X_model_canonicalized)
    feature_engin_X = feature_engin.transform(X_model_canonicalized)

    def testMakeCanonicalizer(self):
        supported_makes = [make.lower() for make in SUPPORTED_MAKES]
        X = self.X_make_canonicalized.query("make != 'other'")
        self.assertTrue(X['make'].str.lower().isin(supported_makes).all())

    def testModelCanonicalizer(self):
        supported_models = [model.lower() for model in SUPPORTED_MODELS]
        X = self.X_model_canonicalized.query("model != 'other'")
        self.assertTrue(X['model'].str.lower().isin(supported_models).all())

    def testMakeCanonicalizerOther(self):
        # we have less than 30 BMW so it should be turned to other 
        make_count = (self.df['make'] == 'BMW').sum()
        self.assertTrue(make_count < 30)
        self.assertTrue(self.X_make_canonicalized['make'].str.lower().str.contains('other').all())
    
    def testModelCanonicalizerOther(self):
        # we have less than 10 Panamera so it should be turned to other 
        model_count = (self.df['model'] == 'Panamera').sum()
        self.assertTrue(model_count < 10)
        self.assertTrue(self.X_model_canonicalized['model'].str.lower().str.contains('other').all())

    def testFeatureEngineering(self):
        created_features = ["car_age", "engine_displacement", "miles_per_year"]
        for feature in created_features:
            self.assertTrue(feature in self.feature_engin_X.columns)

    def testModelTargetEncoding(self):
        model_encoder = ModelTargetEncoder()
        


if __name__ == "__main__":
    unittest.main()