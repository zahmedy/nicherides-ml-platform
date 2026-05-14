# scikit learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor

# import transformers 
from src.features.build_features import (
    MakeCanonicalizer,
    ModelCanonicalizer,
    ModelTargetEncoder,
    FeatureEngineering
)

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

# Full pipeline
def get_model_pipeline(model=ExtraTreesRegressor(n_estimators=1000, random_state=42, n_jobs=-1)):
    model_pipeline = Pipeline([
        ("preprocessing_pipeline", preprocessing_pipeline),
        ("training", model)
    ])
    return model_pipeline

