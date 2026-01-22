
# models/gbm_model.py

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any

def fit_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cfg: Any
) -> Tuple[np.ndarray, Any]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),  # 可选：GBM 不一定需要标准化，但为了统一保持一致
        ("gbm", GradientBoostingRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            random_state=cfg.seed
        ))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_pred, pipe

