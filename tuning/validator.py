"""
Hyperparameter tuning/validation module.
Uses Time Series Cross-Validation to find best parameters for each model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def compute_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation (Rank IC)."""
    if len(y_true) < 3:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


def time_series_cv_score(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3
) -> Tuple[float, float]:
    """
    Evaluate model using Time Series Cross-Validation.
    
    Returns:
        Tuple of (mean_score, std_score)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_cv = X.iloc[train_idx]
        y_train_cv = y.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]
        y_val_cv = y.iloc[val_idx]
        
        try:
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            score = compute_rank_ic(y_val_cv, y_pred_cv)
            scores.append(score)
        except Exception as e:
            scores.append(0.0)
    
    return np.mean(scores), np.std(scores)


# ============================================================
# Model-specific tuning functions
# ============================================================

def tune_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float]:
    """Tune Ridge regression."""
    
    alpha_grid = list(getattr(cfg, 'ridge_alpha_grid', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]))
    n_splits = getattr(cfg, 'tuning_cv_splits', 3)
    
    if verbose:
        print(f"\n  Testing {len(alpha_grid)} alpha values...")
    
    best_score = -np.inf
    best_params = {'alpha': cfg.ridge_alpha}
    
    for alpha in alpha_grid:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        
        mean_score, std_score = time_series_cv_score(model, X_train, y_train, n_splits)
        
        if verbose:
            print(f"    alpha={alpha:<12} → Rank IC: {mean_score:.4f} (±{std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'alpha': alpha}
    
    return best_params, best_score


def tune_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float]:
    """Tune Lasso regression."""
    
    alpha_grid = list(getattr(cfg, 'lasso_alpha_grid', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
    n_splits = getattr(cfg, 'tuning_cv_splits', 3)
    
    if verbose:
        print(f"\n  Testing {len(alpha_grid)} alpha values...")
    
    best_score = -np.inf
    best_params = {'alpha': cfg.lasso_alpha}
    
    for alpha in alpha_grid:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=alpha, max_iter=10000))
        ])
        
        mean_score, std_score = time_series_cv_score(model, X_train, y_train, n_splits)
        
        if verbose:
            print(f"    alpha={alpha:<12} → Rank IC: {mean_score:.4f} (±{std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'alpha': alpha}
    
    return best_params, best_score


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float]:
    """Tune Random Forest with random search."""
    
    n_iter = getattr(cfg, 'tuning_n_iter', 20)
    n_splits = getattr(cfg, 'tuning_cv_splits', 3)
    seed = getattr(cfg, 'seed', 42)
    
    # Parameter options
    n_estimators_opts = list(getattr(cfg, 'rf_n_estimators_grid', [100, 200, 300]))
    max_depth_opts = list(getattr(cfg, 'rf_max_depth_grid', [3, 5, 7, 10, None]))
    min_samples_leaf_opts = list(getattr(cfg, 'rf_min_samples_leaf_grid', [20, 50, 100, 200]))
    
    if verbose:
        print(f"\n  Random search with {n_iter} iterations...")
    
    np.random.seed(seed)
    
    best_score = -np.inf
    best_params = {
        'n_estimators': cfg.n_estimators,
        'max_depth': cfg.max_depth,
        'min_samples_leaf': getattr(cfg, 'min_samples_leaf', 50),
        'random_state': seed,
        'n_jobs': -1
    }
    
    for i in range(n_iter):
        params = {
            'n_estimators': int(np.random.choice(n_estimators_opts)),
            'max_depth': np.random.choice(max_depth_opts),
            'min_samples_leaf': int(np.random.choice(min_samples_leaf_opts)),
            'random_state': seed,
            'n_jobs': -1
        }
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(**params))
        ])
        
        mean_score, std_score = time_series_cv_score(model, X_train, y_train, n_splits)
        
        if verbose:
            print(f"    [{i+1:2d}/{n_iter}] n_est={params['n_estimators']:<3}, "
                  f"depth={str(params['max_depth']):<4}, "
                  f"leaf={params['min_samples_leaf']:<3} "
                  f"→ Rank IC: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, best_score


def tune_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float]:
    """Tune Gradient Boosting with random search."""
    
    n_iter = getattr(cfg, 'tuning_n_iter', 20)
    n_splits = getattr(cfg, 'tuning_cv_splits', 3)
    seed = getattr(cfg, 'seed', 42)
    
    # Parameter options
    n_estimators_opts = list(getattr(cfg, 'gbm_n_estimators_grid', [50, 100, 200]))
    max_depth_opts = list(getattr(cfg, 'gbm_max_depth_grid', [2, 3, 4, 5]))
    learning_rate_opts = list(getattr(cfg, 'gbm_learning_rate_grid', [0.01, 0.05, 0.1]))
    min_samples_leaf_opts = list(getattr(cfg, 'gbm_min_samples_leaf_grid', [50, 100, 200]))
    subsample_opts = list(getattr(cfg, 'gbm_subsample_grid', [0.7, 0.8, 0.9, 1.0]))
    
    if verbose:
        print(f"\n  Random search with {n_iter} iterations...")
    
    np.random.seed(seed)
    
    best_score = -np.inf
    best_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': cfg.learning_rate,
        'min_samples_leaf': 100,
        'subsample': 0.8,
        'random_state': seed
    }
    
    for i in range(n_iter):
        params = {
            'n_estimators': int(np.random.choice(n_estimators_opts)),
            'max_depth': int(np.random.choice(max_depth_opts)),
            'learning_rate': float(np.random.choice(learning_rate_opts)),
            'min_samples_leaf': int(np.random.choice(min_samples_leaf_opts)),
            'subsample': float(np.random.choice(subsample_opts)),
            'random_state': seed
        }
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('gbm', GradientBoostingRegressor(**params))
        ])
        
        mean_score, std_score = time_series_cv_score(model, X_train, y_train, n_splits)
        
        if verbose:
            print(f"    [{i+1:2d}/{n_iter}] lr={params['learning_rate']:.2f}, "
                  f"n_est={params['n_estimators']:<3}, "
                  f"depth={params['max_depth']} "
                  f"→ Rank IC: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, best_score


# ============================================================
# Tuner Registry
# ============================================================

TUNER_REGISTRY = {
    'ridge': tune_ridge,
    'lasso': tune_lasso,
    'random_forest': tune_random_forest,
    'gbm': tune_gbm,
}


# ============================================================
# Main Tuning Function
# ============================================================

def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg,
    model_names: List[str] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Tune all models and return best parameters for each.
    
    Args:
        X_train: Training features
        y_train: Training target
        cfg: Configuration object
        model_names: List of model names to tune (default: all)
        verbose: Whether to print progress
    
    Returns:
        Dictionary: {model_name: {'best_params': {...}, 'best_score': float}}
    """
    
    if model_names is None:
        model_names = ['ridge', 'lasso', 'random_forest', 'gbm']
    
    if verbose:
        print("\n" + "=" * 70)
        print("🔧 PHASE 1: HYPERPARAMETER TUNING (Validation)")
        print("=" * 70)
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Models to tune: {model_names}")
    
    results = {}
    
    for model_name in model_names:
        if model_name not in TUNER_REGISTRY:
            if verbose:
                print(f"\n⚠️ No tuner for {model_name}, skipping...")
            continue
        
        if verbose:
            print(f"\n{'─'*50}")
            print(f"🔧 Tuning: {model_name.upper()}")
            print(f"{'─'*50}")
        
        tuner_fn = TUNER_REGISTRY[model_name]
        best_params, best_score = tuner_fn(X_train, y_train, cfg, verbose)
        
        results[model_name] = {
            'best_params': best_params,
            'best_score': best_score
        }
        
        if verbose:
            print(f"\n  ✅ Best params: {best_params}")
            print(f"  ✅ Best CV score (Rank IC): {best_score:.4f}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("📊 TUNING SUMMARY")
        print("=" * 70)
        print(f"{'Model':<20} {'Best CV Score':>15}")
        print("-" * 40)
        
        for model_name, result in results.items():
            score = result['best_score']
            print(f"{model_name:<20} {score:>15.4f}")
        
        print("-" * 40)
        
        # Best model from tuning
        best_model = max(results.items(), key=lambda x: x[1]['best_score'])
        print(f"\n🏆 Best from tuning: {best_model[0].upper()} (CV Score: {best_model[1]['best_score']:.4f})")
        print("=" * 70)
    
    return results
