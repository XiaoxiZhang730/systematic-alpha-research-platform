# models/random_forest_model.py

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any

def fit_and_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, cfg: Any) -> Tuple[np.ndarray, Any]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.seed,
            n_jobs=-1
        ))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_pred, pipe
