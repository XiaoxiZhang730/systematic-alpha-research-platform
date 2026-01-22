import numpy as np
import pandas as pd
from typing import Dict

def eval_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

def model_rank_ic(panel_test: pd.DataFrame, preds: np.ndarray, target: str = "target") -> Dict[str, float]:
    tmp = panel_test.copy()
    tmp["pred"] = preds
    ric = (
        tmp.groupby(level="date")
           .apply(lambda g: g["pred"].corr(g[target], method="spearman"))
           .dropna()
    )
    if ric.empty:
        return {"mean_rank_ic": np.nan, "rank_ic_ir": np.nan, "n_dates": 0}
    std = ric.std()
    return {
        "mean_rank_ic": float(ric.mean()),
        "rank_ic_ir": float(ric.mean() / std) if std and std > 0 else np.nan,
        "n_dates": int(ric.shape[0]),
    }
