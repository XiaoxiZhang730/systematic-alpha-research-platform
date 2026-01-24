from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def fit_and_predict(X_train, y_train, X_test, cfg, override_params=None):
    """Fit Random Forest model and return predictions."""
    
    # Default params from config
    params = {
        'n_estimators': cfg.n_estimators,
        'max_depth': cfg.max_depth,
        'min_samples_leaf': getattr(cfg, 'min_samples_leaf', 50),
        'random_state': cfg.seed,
        'n_jobs': -1
    }
    
    # Override with tuned params if provided
    if override_params is not None:
        params.update(override_params)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('random_forest', RandomForestRegressor(**params))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred, model
