from dataclasses import dataclass
from typing import Tuple, Set, Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Config:
    # ======================
    # Data
    # ======================
    tickers: Tuple[str, ...] = ("AAPL", "MSFT", "JPM", "XOM", "SPY")
    market: str = "SPY"
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    split_date: pd.Timestamp = pd.Timestamp("2021-12-31")

    # ======================
    # Feature inference rules
    # ======================
    non_feature_cols: Set[str] = frozenset({
        "target", "y", "label", "future_return", "ticker", "date", "ric"
    })
    always_exclude: Set[str] = frozenset({"pred"})
    use_prefix_filter: bool = False
    feature_prefixes: Tuple[str, ...] = ("feat_",)

    # ======================
    # Cross-validation / randomness
    # ======================
    n_splits: int = 5
    seed: int = 42  

    # ======================
    # Linear models (Ridge / Lasso)
    # ======================
    ridge_alpha: float = 1e-3
    lasso_alpha: float = 1e-4
    alpha: float = 1e-4
    alpha_grid: Tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    #alpha: float = 1.0

    # ======================
    # Tree models (RF / GBM)
    # ======================
    n_estimators: int = 200
    max_depth: Optional[int] = None
    learning_rate: float = 0.05  # GBM 

    # ======================
    # Deep learning (MLP)
    # ======================
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 50
