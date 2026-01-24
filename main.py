import pandas as pd
import numpy as np

from configs.config import Config
from data.load import load_ohlcv
from data.preprocess import preprocess_ohlcv
from features.feature_engineering import split_by_date
from tuning.validator import tune_all_models
from execution.multi_model_runner import run_all_models
from models.registry import MODEL_REGISTRY
from portfolio import PortfolioOptimizer, PortfolioConstraints, print_portfolio_summary


def prepare_training_data(cfg):
    """Load and prepare training data for tuning."""
    raw = load_ohlcv(list(cfg.tickers), cfg.start, cfg.end)
    clean = preprocess_ohlcv(raw)
    
    train, test, feature_cols = split_by_date(
        clean,
        split_date=str(cfg.split_date),
        target_window=21
    )
    
    X_train = train[feature_cols]
    y_train = train["target"]
    
    train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    
    return X_train, y_train, train, test, feature_cols


def get_predictions_for_portfolio(cfg, model_name, best_params, test_data, feature_cols):
    """Get model predictions for portfolio construction."""
    
    # Load fresh data
    raw = load_ohlcv(list(cfg.tickers), cfg.start, cfg.end)
    clean = preprocess_ohlcv(raw)
    
    train, test, _ = split_by_date(
        clean,
        split_date=str(cfg.split_date),
        target_window=21
    )
    
    X_train = train[feature_cols]
    y_train = train["target"]
    X_test = test[feature_cols]
    
    # Clean data
    train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    
    test_mask = np.isfinite(X_test).all(axis=1)
    X_test_clean = X_test[test_mask]
    
    # Get model and predict
    model_module = MODEL_REGISTRY[model_name]
    y_pred, model = model_module.fit_and_predict(
        X_train, y_train, X_test_clean, cfg,
        override_params=best_params
    )
    
    # Create predictions DataFrame
    predictions = pd.Series(y_pred, index=X_test_clean.index)
    
    return predictions, test[test_mask]


if __name__ == "__main__":
    
    # ========================================================
    # CONFIGURATION
    # ========================================================
    
    cfg = Config(
        tickers=(
            # Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "ORCL", "ADBE", "CRM", "INTC",
            # Finance
            "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP"
        ),
        market="SPY",
        start="2015-01-01",
        end="2024-12-31",
        split_date=pd.Timestamp("2021-12-31")
    )
    
    print("\n" + "#" * 70)
    print("    SYSTEMATIC PORTFOLIO CONSTRUCTION")
    print("#" * 70)
    print(f"\nTickers: {len(cfg.tickers)} stocks")
    print(f"Date range: {cfg.start} to {cfg.end}")
    print(f"Split date: {cfg.split_date.date()}")
    
    # ========================================================
    # PHASE 1: HYPERPARAMETER TUNING
    # ========================================================
    
    print("\n" + "=" * 70)
    print("🔧 PHASE 1: HYPERPARAMETER TUNING")
    print("=" * 70)
    
    X_train, y_train, train_data, test_data, feature_cols = prepare_training_data(cfg)
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    
    tuning_results = tune_all_models(
        X_train=X_train,
        y_train=y_train,
        cfg=cfg,
        model_names=['ridge', 'lasso', 'random_forest', 'gbm'],
        verbose=True
    )
    
    # ========================================================
    # PHASE 2: TRAIN & TEST
    # ========================================================
    
    print("\n" + "=" * 70)
    print("🧪 PHASE 2: TRAIN & TEST")
    print("=" * 70)
    
    final_results = run_all_models(cfg, tuning_results=tuning_results)
    
    # ========================================================
    # PHASE 3: PORTFOLIO OPTIMIZATION
    # ========================================================
    
    print("\n" + "=" * 70)
    print("💼 PHASE 3: PORTFOLIO OPTIMIZATION")
    print("=" * 70)
    
    # Find best model
    valid_results = [(n, r) for n, r in final_results if r is not None]
    best_model_name, best_result = max(valid_results, key=lambda x: x[1].get('rank_ic_mean', 0))
    best_params = tuning_results[best_model_name]['best_params']
    
    print(f"\nUsing best model: {best_model_name.upper()}")
    print(f"Best params: {best_params}")
    
    # Get predictions
    predictions, test_with_pred = get_predictions_for_portfolio(
        cfg, best_model_name, best_params, test_data, feature_cols
    )
    
    # Get latest predictions (most recent date)
    latest_date = predictions.index.get_level_values('date').max()
    latest_predictions = predictions.xs(latest_date, level='date')
    
    print(f"\nOptimizing portfolio for date: {latest_date.date()}")
    print(f"Stocks with predictions: {len(latest_predictions)}")
    
    # Initialize optimizer
    constraints = PortfolioConstraints(
        max_weight=0.15,        # Max 15% per stock
        min_weight=-0.10,       # Max 10% short
        max_long_weight=1.0,    # 100% long
        max_short_weight=0.3,   # 30% short max
        max_leverage=1.3        # 130% max
    )
    
    optimizer = PortfolioOptimizer(constraints=constraints)
    
    # Try different optimization methods
    methods = ['signal_weighted', 'rank_weighted', 'long_short']
    
    for method in methods:
        print(f"\n{'─'*50}")
        print(f"📊 Method: {method.upper()}")
        print(f"{'─'*50}")
        
        # Optimize
        if method == 'long_short':
            weights = optimizer.optimize(latest_predictions, method=method, n_long=5, n_short=5)
        else:
            weights = optimizer.optimize(latest_predictions, method=method)
        
        # Analyze
        analysis = optimizer.analyze_portfolio(weights, None, latest_predictions)
        print_portfolio_summary(weights, analysis)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    

# import pandas as pd
# from configs.config import Config
# from execution.multi_model_runner import run_all_models 

# if __name__ == "__main__":
#     cfg = Config(
#         tickers=(
#         # ===== Tech =====
#         "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
#         "META", "ORCL", "ADBE", "CRM", "INTC",
#         "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP"
#         ),
#         market="SPY",
#         start="2015-01-01",
#         end="2024-12-31",
#         split_date=pd.Timestamp("2021-12-31")
#     )
#     run_all_models(cfg)

# if __name__ == "__main__":
#     cfg = Config(
#         tickers=(
#         # ===== Tech =====
#         "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
#         "META", "ORCL", "ADBE", "CRM", "INTC",

#         # ===== Financials =====
#         "JPM", "BAC", "GS", "MS", "WFC",
#         "C", "BLK", "AXP",

#         # ===== Healthcare =====
#         "JNJ", "UNH", "PFE", "MRK", "ABBV",
#         "LLY", "TMO",

#         # ===== Consumer Staples =====
#         "PG", "KO", "PEP", "WMT", "COST",

#         # ===== Consumer Discretionary =====
#         "MCD", "HD", "NKE", "LOW", "SBUX",

#         # ===== Industrials =====
#         "CAT", "BA", "GE", "HON", "UPS",

#         # ===== Energy =====
#         "XOM", "CVX",

#         # ===== Communication =====
#         "DIS", "NFLX",

#         # ===== Market / Style ETFs =====
#         "SPY", "QQQ", "IWM"
#         ),
#         market="SPY",
#         start="2015-01-01",
#         end="2024-12-31",
#         split_date=pd.Timestamp("2021-12-31")
#     )
#     run_all_models(cfg)


# import pandas as pd
# import numpy as np

# from configs.config import Config
# from data.load import load_ohlcv
# from data.preprocess import preprocess_ohlcv
# from features.feature_engineering import split_by_date
# from tuning.validator import tune_all_models
# from execution.multi_model_runner import run_all_models


# def prepare_training_data(cfg):
#     """
#     Load and prepare training data for tuning.
    
#     Returns:
#         X_train, y_train: Clean training data
#     """
#     # Load data
#     raw = load_ohlcv(list(cfg.tickers), cfg.start, cfg.end)
#     clean = preprocess_ohlcv(raw)
    
#     # Split data
#     train, test, feature_cols = split_by_date(
#         clean,
#         split_date=str(cfg.split_date),
#         target_window=21
#     )
    
#     # Prepare training data
#     X_train = train[feature_cols]
#     y_train = train["target"]
    
#     # Clean NaN/Inf
#     train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
#     X_train = X_train[train_mask]
#     y_train = y_train[train_mask]
    
#     return X_train, y_train


# if __name__ == "__main__":
    
#     # ========================================================
#     # CONFIGURATION
#     # ========================================================
    
#     cfg = Config(
#         tickers=(
#             # ===== Tech =====
#             "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
#             "META", "ORCL", "ADBE", "CRM", "INTC",
#             # ===== Finance =====
#             "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP"
#         ),
#         market="SPY",
#         start="2015-01-01",
#         end="2024-12-31",
#         split_date=pd.Timestamp("2021-12-31")
#     )
    
#     print("\n" + "#" * 70)
#     print("    SYSTEMATIC PORTFOLIO CONSTRUCTION")
#     print("#" * 70)
#     print(f"\nTickers: {len(cfg.tickers)} stocks")
#     print(f"Date range: {cfg.start} to {cfg.end}")
#     print(f"Split date: {cfg.split_date.date()}")
    
#     # ========================================================
#     # OPTION 1: RUN WITHOUT TUNING (Fast - like before)
#     # ========================================================
    
#     # Uncomment this to run without tuning:
#     # run_all_models(cfg)
    
#     # ========================================================
#     # OPTION 2: RUN WITH TUNING (Recommended)
#     # ========================================================
    
#     # --- PHASE 1: TUNING ---
#     print("\n" + "=" * 70)
#     print("🔧 PHASE 1: HYPERPARAMETER TUNING")
#     print("=" * 70)
    
#     # Prepare training data
#     print("\nPreparing training data...")
#     X_train, y_train = prepare_training_data(cfg)
#     print(f"Training samples: {len(X_train)}")
#     print(f"Features: {X_train.shape[1]}")
    
#     # Tune all models
#     tuning_results = tune_all_models(
#         X_train=X_train,
#         y_train=y_train,
#         cfg=cfg,
#         model_names=['ridge', 'lasso', 'random_forest', 'gbm'],
#         verbose=True
#     )
    
#     # --- PHASE 2: TRAIN & TEST ---
#     print("\n" + "=" * 70)
#     print("🧪 PHASE 2: TRAIN & TEST (with tuned parameters)")
#     print("=" * 70)
    
#     # Run all models with tuned parameters
#     final_results = run_all_models(cfg, tuning_results=tuning_results)
