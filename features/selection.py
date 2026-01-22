import numpy as np
import pandas as pd
from typing import Set, Tuple, List, Dict, Iterator

def infer_feature_columns(df: pd.DataFrame,
                          non_feature_cols: Set[str],
                          always_exclude: Set[str],
                          use_prefix_filter: bool = False,
                          feature_prefixes: Tuple[str, ...] = ("feat_",)) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    candidates = [c for c in numeric_cols if c not in non_feature_cols and c not in always_exclude]
    if use_prefix_filter:
        candidates = [c for c in candidates if any(c.startswith(p) for p in feature_prefixes)]
    return candidates

def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if tmp.shape[0] < 3 or tmp.nunique().min() <= 1:
        return np.nan
    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method=method)

def compute_ic_series(panel: pd.DataFrame, feature: str, target: str = "target", method: str = "pearson") -> pd.Series:
    return (panel.groupby(level="date")
                 .apply(lambda g: _safe_corr(g[feature], g[target], method=method))
                 .dropna())

def summarize_ic(ic: pd.Series) -> Dict[str, float]:
    if ic.empty:
        return {"mean": np.nan, "std": np.nan, "ir": np.nan, "pos_ratio": np.nan, "n_dates": 0}
    std = ic.std()
    return {
        "mean": float(ic.mean()),
        "std": float(std),
        "ir": float(ic.mean() / std) if std and std > 0 else np.nan,
        "pos_ratio": float((ic > 0).mean()),
        "n_dates": int(ic.shape[0]),
    }

def iter_rankic_rows(train: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str], target: str = "target") -> Iterator[Dict[str, float]]:
    for f in feature_cols:
        ric_tr = compute_ic_series(train, f, target=target, method="spearman")
        ric_te = compute_ic_series(test,  f, target=target, method="spearman")
        trr = summarize_ic(ric_tr)
        ter = summarize_ic(ric_te)
        yield {
            "feature": f,
            "train_RankIC_mean": trr["mean"],
            "train_RankIC_IR": trr["ir"],
            "train_RankIC_pos_ratio": trr["pos_ratio"],
            "test_RankIC_mean": ter["mean"],
            "test_RankIC_IR": ter["ir"],
            "test_RankIC_pos_ratio": ter["pos_ratio"],
        }

def select_features_by_ic_stream(
    rows: Iterator[Dict[str, float]],
    min_rank_ic: float = 0.01,
    min_ir: float = 0.3,
    min_pos_ratio: float = 0.55,
    use_train: bool = True,
) -> List[str]:
    prefix = "train_" if use_train else "test_"
    out = []
    for r in rows:
        m = r.get(f"{prefix}RankIC_mean", np.nan)
        ir = r.get(f"{prefix}RankIC_IR", np.nan)
        pr = r.get(f"{prefix}RankIC_pos_ratio", np.nan)
        if np.isfinite(m) and np.isfinite(ir) and np.isfinite(pr) and abs(m) >= min_rank_ic and ir >= min_ir and pr >= min_pos_ratio:
            out.append(r["feature"])
    return out
