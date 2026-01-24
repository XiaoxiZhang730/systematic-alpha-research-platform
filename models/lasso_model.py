from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def fit_and_predict(X_train, y_train, X_test, cfg, override_params=None):
    """Fit Lasso model and return predictions."""
    
    # Use override params if provided, else use config
    if override_params is not None and 'alpha' in override_params:
        alpha = override_params['alpha']
    else:
        alpha = cfg.lasso_alpha
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=alpha, max_iter=10000))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred, model
