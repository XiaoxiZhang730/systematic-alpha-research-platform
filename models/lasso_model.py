# # models/lasso_model.py

# import numpy as np
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LassoCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import TimeSeriesSplit
# from typing import Tuple, Any

# def fit_and_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, cfg: Any) -> Tuple[np.ndarray, Any]:
#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("lasso", LassoCV(alphas=np.array(cfg.alphas), cv=TimeSeriesSplit(n_splits=cfg.n_splits), random_state=cfg.seed))
#     ])
#     pipe.fit(X_train, y_train)
#     y_pred = pipe.predict(X_test)
#     return y_pred, pipe

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def fit_and_predict(X_train, y_train, X_test, cfg):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=cfg.lasso_alpha))
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test), pipe