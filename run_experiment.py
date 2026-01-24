"""
Run single model experiment with optional pre-tuned parameters.
"""

import numpy as np
import pandas as pd

from data.load import load_ohlcv
from data.preprocess import preprocess_ohlcv
from features.feature_engineering import split_by_date
from models.registry import MODEL_REGISTRY
from evaluation.metrics import eval_regression
from evaluation.validation import validate_predictions


def run_experiment(cfg, model_name: str = "ridge", best_params=None):
    """
    Improved experiment runner with validation.
    
    Args:
        cfg: Configuration object
        model_name: Name of the model to run
        best_params: Pre-tuned hyperparameters (optional)
    
    Returns:
        Dictionary with results
    """
    
    # ===== LOAD AND PREPROCESS =====
    raw = load_ohlcv(list(cfg.tickers), cfg.start, cfg.end)
    clean = preprocess_ohlcv(raw)
    
    # ===== PREPARE DATA =====
    train, test, feature_cols = split_by_date(
        clean, 
        split_date=str(cfg.split_date),
        target_window=21  # Predict 1-month forward return
    )
    
    # ===== CLEAN DATA =====
    X_train = train[feature_cols]
    y_train = train["target"]
    X_test = test[feature_cols]
    y_test = test["target"]
    
    # Remove any remaining NaN/Inf
    train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    
    test_mask = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # ===== TRAIN MODEL (with optional tuned params) =====
    model_module = MODEL_REGISTRY[model_name]
    
    # Pass best_params to the model's fit_and_predict function
    y_pred, model_obj = model_module.fit_and_predict(
        X_train, y_train, X_test, cfg,
        override_params=best_params  # <-- Pass tuned params here
    )
    
    # ===== VALIDATION =====
    validation_results = validate_predictions(y_test, y_pred, model_name)
    
    # ===== EVALUATE =====
    test_metrics = eval_regression(y_test, y_pred)
    
    # ===== COMPUTE RANK IC =====
    test_results = test.loc[test_mask].copy()
    test_results["pred"] = y_pred
    
    rank_ic_by_date = test_results.groupby("date").apply(
        lambda x: x["target"].corr(x["pred"], method="spearman")
    ).dropna()
    
    # ===== PRINT RESULTS =====
    print(f"\n=== Results ===")
    print(f"Mean Rank IC: {rank_ic_by_date.mean():.4f}")
    print(f"Rank IC Std: {rank_ic_by_date.std():.4f}")
    print(f"Rank IC IR: {rank_ic_by_date.mean() / rank_ic_by_date.std():.4f}")
    print(f"Hit Rate (IC > 0): {(rank_ic_by_date > 0).mean():.2%}")
    print(f"R2: {test_metrics['r2']:.4f}")
    
    # ===== PRINT VALIDATION =====
    print(f"\n=== Validation ===")
    print(f"Spearman Corr: {validation_results.get('spearman_corr', float('nan')):.4f}")
    print(f"Long-Short Spread: {validation_results.get('long_short_spread', float('nan')):.4f}")
    print(f"Monotonic: {'✅ Yes' if validation_results.get('is_monotonic') else '❌ No'}")
    
    # Print quintile returns
    quintiles = validation_results.get('quintile_returns', {})
    if quintiles:
        print(f"\nQuintile Returns (Actual returns by predicted quintile):")
        for q, ret in sorted(quintiles.items()):
            bar = "█" * int(abs(ret) * 500)
            sign = "+" if ret >= 0 else ""
            print(f"  {q}: {sign}{ret:.4f} {bar}")
    
    # ===== RETURN OUTPUT =====
    return {
        "model": model_name,
        "best_params": best_params,
        "rank_ic_mean": float(rank_ic_by_date.mean()),
        "rank_ic_std": float(rank_ic_by_date.std()),
        "rank_ic_ir": float(rank_ic_by_date.mean() / rank_ic_by_date.std()),
        "hit_rate": float((rank_ic_by_date > 0).mean()),
        "metrics": test_metrics,
        "validation": validation_results,
        "used_features": feature_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
