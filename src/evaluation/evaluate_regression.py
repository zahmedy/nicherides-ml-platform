from sklearn.metrics import mean_absolute_error, r2_score

def get_mae_and_r2(y_pred, y_test):
    return (mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred))
