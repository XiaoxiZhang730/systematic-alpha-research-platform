from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def fit_and_predict(X_train, y_train, X_test, cfg, override_params=None):
    """Fit GBM model and return predictions."""
    
    # Default params from config
    params = {
        'n_estimators': getattr(cfg, 'gbm_n_estimators', 100),
        'max_depth': getattr(cfg, 'gbm_max_depth', 3),
        'learning_rate': cfg.learning_rate,
        'min_samples_leaf': getattr(cfg, 'gbm_min_samples_leaf', 100),
        'subsample': getattr(cfg, 'gbm_subsample', 0.8),
        'random_state': cfg.seed
    }
    
    # Override with tuned params if provided
    if override_params is not None:
        params.update(override_params)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('gbm', GradientBoostingRegressor(**params))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred, model

