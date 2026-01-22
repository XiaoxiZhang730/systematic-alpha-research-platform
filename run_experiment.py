import os
import json
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path

from configs.config import Config
from data.load import load_ohlcv
from data.preprocess import preprocess_ohlcv
from features.feature_engineering import make_features, clean_features, add_target, split_by_date
from features.selection import infer_feature_columns, iter_rankic_rows, select_features_by_ic_stream
from models.registry import MODEL_REGISTRY
from utils.hashing import generate_run_id
from utils.metrics import eval_regression, model_rank_ic
from utils.serialization import json_safe


# def run_experiment(cfg: Config, model_name: str = "ridge"):
#     #print(f"\n Running experiment with model: {model_name}")

#     # Step A: Load + Clean Data
#     raw = load_ohlcv(list(cfg.tickers), cfg.start, cfg.end)
#     clean = preprocess_ohlcv(raw)

#     # Step B: Features
#     feat = make_features(clean, market=cfg.market)
#     feat = clean_features(feat)
#     panel = add_target(feat)

#     # Step C: Train/Test Split
#     tr, te = split_by_date(panel, cfg.split_date)

#     # Step D: Feature inference & selection
#     feature_cols = infer_feature_columns(
#         panel,
#         non_feature_cols=cfg.non_feature_cols,
#         always_exclude=cfg.always_exclude,
#         use_prefix_filter=cfg.use_prefix_filter,
#         feature_prefixes=cfg.feature_prefixes,
#     )
#     rows = iter_rankic_rows(tr, te, feature_cols)
#     selected = select_features_by_ic_stream(rows)
#     used_features = selected if selected else feature_cols

#     # Step E: Prepare training data
#     X_train, y_train = tr[used_features], tr["target"]
#     X_test, y_test = te[used_features], te["target"]

#     mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
#     X_train, y_train = X_train.loc[mask], y_train.loc[mask]

#     n_train_raw = len(tr)
#     n_train_kept = len(X_train)
#     n_test = len(X_test)
#     n_tickers_train = tr.index.get_level_values(0).nunique()
#     n_tickers_test = te.index.get_level_values(0).nunique()

#     # Step F: Model fitting & prediction
#     model_module = MODEL_REGISTRY[model_name]
#     y_pred, model_obj = model_module.fit_and_predict(X_train, y_train, X_test, cfg)

#     # Step G: Evaluation
#     test_metrics = eval_regression(y_test, y_pred)
#     train_metrics = eval_regression(y_train, model_obj.predict(X_train))
#     ric = model_rank_ic(te, y_pred)

#     # Step G.1: Coefficients (if linear model)
#     coefs = None
#     coef_stats = {}
#     top_coef = {}

#     if hasattr(model_obj.named_steps.get(model_name), "coef_"):
#         coefs = pd.Series(model_obj.named_steps[model_name].coef_, index=used_features)
#         coef_abs = coefs.abs()
#         coef_stats = {
#             "mean": coef_abs.mean(),
#             "median": coef_abs.median(),
#             "max": coef_abs.max()
#         }
#         top_coef = coef_abs.sort_values(ascending=False).head(10).to_dict()

#     # Step G.2: Model-specific fields
#     model_specific_fields = {}
#     if model_name in {"ridge", "lasso"}:
#         best_alpha = getattr(model_obj.named_steps[model_name], "alpha_", None)
#         model_specific_fields["best_alpha"] = best_alpha

#     # Step H: Save output
#     run_id = generate_run_id(cfg, used_features, model_name=model_name)
#     model_params = model_obj.get_params(deep=True)

#     output = {
#         "run_id": run_id,
#         "model": model_name,
#         "config": {k: json_safe(getattr(cfg, k)) for k in cfg.__annotations__},
#         "model_params": json_safe(model_params),
#         "metrics": test_metrics,
#         "train_metrics": train_metrics,
#         "rank_ic": ric,
#         "selected_features": selected,
#         "used_features": used_features,
#         "n_train_raw": n_train_raw,
#         "n_train_kept": n_train_kept,
#         "n_test": n_test,
#         "n_tickers_train": n_tickers_train,
#         "n_tickers_test": n_tickers_test,
#         "coef_stats": coef_stats,
#         "top_coef": top_coef
#     }

#     output.update(model_specific_fields)

#     Path("runs").mkdir(exist_ok=True)
#     with open(f"runs/{run_id}.json", "w") as f:
#         json.dump(output, f, indent=2, default=json_safe)

#     # Step I: Print summary
#     print("\n=== Experiment Summary ===")
#     print(f"Run ID          : {run_id}")
#     print(f"Date Range      : {cfg.start} → {cfg.end} (Split: {cfg.split_date})")
#     print(f"Sample Size     : Train = {n_train_raw:,} → {n_train_kept:,} (kept {n_train_kept / n_train_raw:.1%}), Test = {n_test:,}")
#     print(f"Tickers Covered : Train = {n_tickers_train}, Test = {n_tickers_test}")
#     print(f"Model           : {model_name}")
#     if "best_alpha" in model_specific_fields:
#         print(f"Best Alpha (λ)  : {model_specific_fields['best_alpha']:.4g}")
#     print(f"Features Used   : {len(used_features)} (Selected: {len(selected) if selected else len(used_features)})")

#     if coefs is not None:
#         print(f"Coef Stats      : Mean = {coef_stats['mean']:.4g}, Median = {coef_stats['median']:.4g}, Max = {coef_stats['max']:.4g}")
#         print("\nTop 10 Features by |coef|:")
#         for feat, val in top_coef.items():
#             print(f"  {feat:<30} {val:.4g}")

#     print("\n--- Performance ---")
#     print(f"Train R2        : {train_metrics['r2']:.4f}")
#     print(f"Train MAE       : {train_metrics['mae']:.4f}")
#     print(f"Test R2         : {test_metrics['r2']:.4f}")
#     print(f"Test MAE        : {test_metrics['mae']:.4f}")
#     print(f"Test RMSE       : {test_metrics['rmse']:.4f}")
#     print(f"RankIC (mean)   : {ric['mean_rank_ic']:.4f}")
#     print("===========================\n")

#     return output 

def run_experiment(cfg: Config, model_name: str = "ridge"):
    raw = load_ohlcv(list(cfg.tickers), cfg.start, cfg.end)
    clean = preprocess_ohlcv(raw)

    feat = make_features(clean, market=cfg.market)
    feat = clean_features(feat)
    # feat = feat.groupby("ticker").shift(1)

    panel = add_target(feat)

    tr, te = split_by_date(panel, cfg.split_date)

    feature_cols = infer_feature_columns(
        panel,
        non_feature_cols=cfg.non_feature_cols,
        always_exclude=cfg.always_exclude,
        use_prefix_filter=cfg.use_prefix_filter,
        feature_prefixes=cfg.feature_prefixes,
    )
    rows = iter_rankic_rows(tr, te, feature_cols)
    selected = select_features_by_ic_stream(rows)
    used_features = selected if selected else feature_cols

    X_train, y_train = tr[used_features], tr["target"]
    X_test, y_test = te[used_features], te["target"]

    mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train, y_train = X_train.loc[mask], y_train.loc[mask]

    n_train_raw = len(tr)
    n_train_kept = len(X_train)
    n_test = len(X_test)
    n_tickers_train = tr.index.get_level_values(0).nunique()
    n_tickers_test = te.index.get_level_values(0).nunique()

    model_module = MODEL_REGISTRY[model_name]
    y_pred, model_obj = model_module.fit_and_predict(X_train, y_train, X_test, cfg)

    test_metrics = eval_regression(y_test, y_pred)
    train_metrics = eval_regression(y_train, model_obj.predict(X_train))
    ric = model_rank_ic(te, y_pred)

    coefs = None
    coef_stats = {}
    top_coef = {}

    if hasattr(model_obj.named_steps.get(model_name), "coef_"):
        coefs = pd.Series(model_obj.named_steps[model_name].coef_, index=used_features)
        coef_abs = coefs.abs()
        coef_stats = {
            "mean": coef_abs.mean(),
            "median": coef_abs.median(),
            "max": coef_abs.max()
        }
        top_coef = coef_abs.sort_values(ascending=False).head(10).to_dict()

    # This dict can now be empty or skipped
    model_specific_fields = {}

    run_id = generate_run_id(cfg, used_features, model_name=model_name)
    model_params = model_obj.get_params(deep=True)

    output = {
        "run_id": run_id,
        "model": model_name,
        "config": {k: json_safe(getattr(cfg, k)) for k in cfg.__annotations__},
        "model_params": json_safe(model_params),
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "rank_ic": ric,
        "selected_features": selected,
        "used_features": used_features,
        "n_train_raw": n_train_raw,
        "n_train_kept": n_train_kept,
        "n_test": n_test,
        "n_tickers_train": n_tickers_train,
        "n_tickers_test": n_tickers_test,
        "coef_stats": coef_stats,
        "top_coef": top_coef
    }

    output.update(model_specific_fields)

    Path("runs").mkdir(exist_ok=True)
    with open(f"runs/{run_id}.json", "w") as f:
        json.dump(output, f, indent=2, default=json_safe)

    print("\n=== Experiment Summary ===")
    print(f"Run ID          : {run_id}")
    print(f"Date Range      : {cfg.start} → {cfg.end} (Split: {cfg.split_date})")
    print(f"Sample Size     : Train = {n_train_raw:,} → {n_train_kept:,} (kept {n_train_kept / n_train_raw:.1%}), Test = {n_test:,}")
    print(f"Tickers Covered : Train = {n_tickers_train}, Test = {n_tickers_test}")
    print(f"Model           : {model_name}")
    print(f"Alpha Used      : {cfg.ridge_alpha}")
    print(f"Features Used   : {len(used_features)} (Selected: {len(selected) if selected else len(used_features)})")

    if coefs is not None:
        print(f"Coef Stats      : Mean = {coef_stats['mean']:.4g}, Median = {coef_stats['median']:.4g}, Max = {coef_stats['max']:.4g}")
        print("\nTop 10 Features by |coef|:")
        for feat, val in top_coef.items():
            print(f"  {feat:<30} {val:.4g}")

    print("\n--- Performance ---")
    print(f"Train R2        : {train_metrics['r2']:.4f}")
    print(f"Train MAE       : {train_metrics['mae']:.4f}")
    print(f"Test R2         : {test_metrics['r2']:.4f}")
    print(f"Test MAE        : {test_metrics['mae']:.4f}")
    print(f"Test RMSE       : {test_metrics['rmse']:.4f}")
    print(f"RankIC (mean)   : {ric['mean_rank_ic']:.4f}")
    print("===========================\n")

    return output
