# scikit learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor

# import transformers 
from src.features.build_features import (
    MakeCanonicalizer,
    ModelCanonicalizer,
    ModelTargetEncoder,
    FeatureEngineering
)
from src.data.clean_data import drop_more_than_4_nans, replace_low_price_with_avg
from src.data.validate_data import filter_bad_rows, filter_columns

categorical_features = ['make', 'color', 'body_type', 
                        'fuel_type', 'drivetrain', 'transmission']
numeric_features = ['model', 'year', 'engine_volume', 
                    'mileage', 'engine_cylinders']

# Preprocessing numrical features pipeline
numeric_pipeline = Pipeline([
    ("FeatureEngineering", FeatureEngineering()),
    ("ModelCanonicalizer", ModelCanonicalizer()),
    ("ModelTargetEncoder", ModelTargetEncoder())
])

# Preprocessing categroical features pipeline
categorical_pipeline = Pipeline([
    ("MakeCanonicalizer", MakeCanonicalizer()),
    ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessing transformer pipleine
preprocessing_pipeline = ColumnTransformer([
    ("numeric_pipeline", numeric_pipeline, numeric_features),
    ("categorical_pipeline", categorical_pipeline, categorical_features)
])


def get_data_quality_pipeline():
    return Pipeline([
        ("replace_low_price_with_avg", FunctionTransformer(replace_low_price_with_avg)),
        ("filter_columns", FunctionTransformer(filter_columns)),
        ("filter_bad_rows", FunctionTransformer(filter_bad_rows)),
        ("drop_more_than_4_nans", FunctionTransformer(drop_more_than_4_nans)),
    ])


# Full pipeline
def get_model_pipeline(model=ExtraTreesRegressor(n_estimators=1000, random_state=42, n_jobs=-1)):
    model_pipeline = Pipeline([
        ("preprocessing_pipeline", preprocessing_pipeline),
        ("training", model)
    ])
    return model_pipeline
