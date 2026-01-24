from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def fit_and_predict(X_train, y_train, X_test, cfg, override_params=None):
    """
    Fit Ridge model and return predictions.
    
    Args:
        X_train, y_train: Training data
        X_test: Test features
        cfg: Config object
        override_params: Optional dict of tuned parameters
    """
    
    # Use override params if provided, else use config
    if override_params is not None and 'alpha' in override_params:
        alpha = override_params['alpha']
    else:
        alpha = cfg.ridge_alpha
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred, model

