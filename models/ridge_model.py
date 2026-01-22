# import numpy as np
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import RidgeCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import TimeSeriesSplit
# from typing import Tuple, Any

# def fit_and_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, cfg: Any) -> Tuple[np.ndarray, Any]:
#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("ridge", RidgeCV(alphas=np.array(cfg.alphas), cv=TimeSeriesSplit(n_splits=cfg.n_splits), scoring="r2"))
#     ])
#     pipe.fit(X_train, y_train)
#     y_pred = pipe.predict(X_test)
#     return y_pred, pipe

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def fit_and_predict(X_train, y_train, X_test, cfg):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=cfg.ridge_alpha))
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test), pipe
