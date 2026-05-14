from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score


def get_cross_val_scores(model_pipeline, X, y):
    scores = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error",
        error_score="raise"
    )

    mae_scores = -scores
    return mae_scores

def get_basic_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return mae, rmse, r2, mape

    
