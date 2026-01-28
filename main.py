import pandas as pd
import numpy as np

from configs.config import Config
from data.load import load_ohlcv
from data.preprocess import preprocess_ohlcv
from features.feature_engineering import split_by_date
from tuning.validator import tune_all_models
from execution.multi_model_runner import run_all_models
from models.registry import MODEL_REGISTRY
from portfolio import (
    PortfolioOptimizer, 
    PortfolioConstraints, 
    print_portfolio_summary,
    Backtester,
    print_backtest_comparison
)
from risk import (
    PostTradeRiskManager,
    print_daily_risk_report,
    print_risk_alerts,
    print_risk_summary
)


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
    
    return X_train, y_train, train, test, feature_cols, clean


def get_predictions_for_portfolio(cfg, model_name, best_params, feature_cols):
    """Get model predictions for portfolio construction."""
    
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
    
    train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    
    test_mask = np.isfinite(X_test).all(axis=1)
    X_test_clean = X_test[test_mask]
    
    model_module = MODEL_REGISTRY[model_name]
    y_pred, model = model_module.fit_and_predict(
        X_train, y_train, X_test_clean, cfg,
        override_params=best_params
    )
    
    predictions = pd.Series(y_pred, index=X_test_clean.index)
    
    return predictions, test[test_mask]


def get_returns_for_backtest(cfg):
    """Get daily returns for backtesting."""
    
    raw = load_ohlcv(list(cfg.tickers), cfg.start, cfg.end)
    clean = preprocess_ohlcv(raw)
    
    if 'ret_1d' in clean.columns:
        returns = clean.pivot_table(
            index='date', 
            columns='ticker', 
            values='ret_1d'
        )
    else:
        prices = clean.pivot_table(
            index='date',
            columns='ticker',
            values='Close'
        )
        returns = prices.pct_change()
    
    return returns.dropna()


def get_prices_for_risk(cfg):
    """Get daily prices for risk management."""
    
    raw = load_ohlcv(list(cfg.tickers), cfg.start, cfg.end)
    clean = preprocess_ohlcv(raw)
    
    prices = clean.pivot_table(
        index='date',
        columns='ticker',
        values='Close'
    )
    
    return prices


if __name__ == "__main__":
    
    # ========================================================
    # CONFIGURATION
    # ========================================================
    
    cfg = Config(
        tickers = (
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
    
    X_train, y_train, train_data, test_data, feature_cols, clean_data = prepare_training_data(cfg)
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
    
    # Find best model
    valid_results = [(n, r) for n, r in final_results if r is not None]
    best_model_name, best_result = max(valid_results, key=lambda x: x[1].get('rank_ic_mean', 0))
    best_params = tuning_results[best_model_name]['best_params']
    
    print(f"\n🏆 Best Model: {best_model_name.upper()}")
    print(f"   Rank IC: {best_result['rank_ic_mean']:.4f}")
    print(f"   Best Params: {best_params}")
    
    # ========================================================
    # PHASE 3: PORTFOLIO OPTIMIZATION
    # ========================================================
    
    print("\n" + "=" * 70)
    print("💼 PHASE 3: PORTFOLIO OPTIMIZATION")
    print("=" * 70)
    
    predictions, test_with_pred = get_predictions_for_portfolio(
        cfg, best_model_name, best_params, feature_cols
    )
    
    latest_date = predictions.index.get_level_values('date').max()
    latest_predictions = predictions.xs(latest_date, level='date')
    
    print(f"\nOptimizing portfolio for date: {latest_date.date()}")
    print(f"Stocks with predictions: {len(latest_predictions)}")
    
    constraints = PortfolioConstraints(
        max_weight=0.15,
        min_weight=-0.10,
        max_long_weight=1.0,
        max_short_weight=0.3,
        max_leverage=1.3
    )
    
    optimizer = PortfolioOptimizer(constraints=constraints)
    
    methods = ['signal_weighted', 'rank_weighted', 'long_short']
    
    for method in methods:
        print(f"\n{'─'*50}")
        print(f"📊 Method: {method.upper()}")
        print(f"{'─'*50}")
        
        if method == 'long_short':
            weights = optimizer.optimize(latest_predictions, method=method, n_long=5, n_short=5)
        else:
            weights = optimizer.optimize(latest_predictions, method=method)
        
        analysis = optimizer.analyze_portfolio(weights, None, latest_predictions)
        print_portfolio_summary(weights, analysis)
    
    # ========================================================
    # PHASE 4: BACKTESTING
    # ========================================================
    
    print("\n" + "=" * 70)
    print("📈 PHASE 4: BACKTESTING")
    print("=" * 70)
    
    print("\nPreparing data for backtest...")
    returns = get_returns_for_backtest(cfg)
    
    test_start = cfg.split_date
    returns_test = returns[returns.index > test_start]
    
    print(f"Backtest period: {returns_test.index[0].date()} to {returns_test.index[-1].date()}")
    print(f"Trading days: {len(returns_test)}")
    
    backtester = Backtester(
        optimizer=optimizer,
        rebalance_freq='M',
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    backtest_results = {}
    
    print("\n" + "─" * 70)
    results_signal = backtester.run(
        predictions=predictions,
        returns=returns_test,
        method='signal_weighted'
    )
    backtest_results['Signal Weighted'] = results_signal
    
    print("\n" + "─" * 70)
    results_rank = backtester.run(
        predictions=predictions,
        returns=returns_test,
        method='rank_weighted'
    )
    backtest_results['Rank Weighted'] = results_rank
    
    print("\n" + "─" * 70)
    results_ls = backtester.run(
        predictions=predictions,
        returns=returns_test,
        method='long_short',
        n_long=5,
        n_short=5
    )
    backtest_results['Long-Short (5/5)'] = results_ls
    
    print("\n" + "─" * 70)
    results_equal = backtester.run(
        predictions=predictions,
        returns=returns_test,
        method='equal_weight'
    )
    backtest_results['Equal Weight'] = results_equal
    
    # ========================================================
    # PHASE 5: COMPARISON & SELECT BEST STRATEGY
    # ========================================================
    
    print("\n" + "=" * 70)
    print("🏆 PHASE 5: FINAL COMPARISON")
    print("=" * 70)
    
    print_backtest_comparison(backtest_results)
    
    best_strategy_name = max(backtest_results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
    best_strategy_results = backtest_results[best_strategy_name]
    
    # Get final weights based on best strategy
    if best_strategy_name == 'Signal Weighted':
        final_weights = optimizer.optimize(latest_predictions, method='signal_weighted')
    elif best_strategy_name == 'Rank Weighted':
        final_weights = optimizer.optimize(latest_predictions, method='rank_weighted')
    elif 'Long-Short' in best_strategy_name:
        final_weights = optimizer.optimize(latest_predictions, method='long_short', n_long=5, n_short=5)
    else:
        final_weights = optimizer.optimize(latest_predictions, method='equal_weight')
    
    # ========================================================
    # PHASE 6: POST-TRADE RISK MANAGEMENT
    # ========================================================
    
    print("\n" + "=" * 70)
    print("🛡️ PHASE 6: POST-TRADE RISK MANAGEMENT")
    print("=" * 70)
    
    # Get prices for risk management
    prices = get_prices_for_risk(cfg)
    prices_test = prices[prices.index > test_start]

    # Simulate portfolio over test period
    starting_capital = 1_000_000  # $1M starting capital
    portfolio_value = starting_capital
    
    # Entry prices (first day prices)
    entry_prices = prices_test.iloc[0]
    
    # Get dates from backtest period
    backtest_dates = returns_test.index.tolist()

    # Initialize post-trade risk manager
    risk_manager = PostTradeRiskManager(
        max_drawdown=0.15,           # 15% max drawdown limit
        position_stop_loss=0.10,     # 10% stop-loss per position
        portfolio_stop_loss=0.05,    # 5% daily portfolio stop-loss
        var_limit_95=0.03,           # 3% VaR limit
        volatility_lookback=21,       # 21-day rolling volatility
        initial_value=starting_capital 
    )
    
    print("\n📊 Simulating post-trade risk monitoring...")
    print(f"   Max Drawdown Limit: {risk_manager.max_drawdown:.0%}")
    print(f"   Position Stop-Loss: {risk_manager.position_stop_loss:.0%}")
    print(f"   Portfolio Stop-Loss: {risk_manager.portfolio_stop_loss:.0%}")
    print(f"   VaR Limit (95%): {risk_manager.var_limit_95:.0%}")
    
    
    # Track portfolio value over time
    print(f"\n   Monitoring {len(backtest_dates)} trading days...")
    
    significant_events = []
    
    for i, date in enumerate(backtest_dates):
        # Get daily returns
        if date in returns_test.index:
            daily_returns = returns_test.loc[date]
            
            # Calculate portfolio return (weighted average of stock returns)
            available_tickers = [t for t in final_weights.index if t in daily_returns.index]
            if len(available_tickers) > 0:
                weights_aligned = final_weights[available_tickers]
                portfolio_return = (weights_aligned * daily_returns[available_tickers]).sum()
                portfolio_value = portfolio_value * (1 + portfolio_return)
        
        # Get current prices
        current_prices = prices_test.loc[date] if date in prices_test.index else None
        
        # Update risk manager
        report = risk_manager.update(
            date=date,
            portfolio_value=portfolio_value,
            weights=final_weights,
            prices=current_prices,
            entry_prices=entry_prices
        )
        
        # Track significant events (alerts or big moves)
        if report.alerts or abs(report.daily_return) > 0.03:
            significant_events.append(report)
        
        # Check if stopped out
        if risk_manager.is_stopped_out:
            print(f"\n   🛑 STOPPED OUT on {date.date()} - Exiting all positions")
            break
    
    # Print significant events (limit to last 5)
    if significant_events:
        print(f"\n📋 Significant Events (showing last {min(5, len(significant_events))}):")
        for report in significant_events[-5:]:
            print_daily_risk_report(report)
    
    # Print final risk summary
    risk_summary = risk_manager.get_risk_summary()
    print_risk_summary(risk_summary)
    
    # Print all alerts
    all_alerts = risk_manager.get_alerts()
    critical_alerts = [a for a in all_alerts if a.level.value == "CRITICAL"]
    warning_alerts = [a for a in all_alerts if a.level.value == "WARNING"]
    
    print(f"\n📋 Alert Summary:")
    print(f"   Total Alerts: {len(all_alerts)}")
    print(f"   Critical: {len(critical_alerts)}")
    print(f"   Warnings: {len(warning_alerts)}")
    
    if critical_alerts:
        print(f"\n🔴 Critical Alerts:")
        print_risk_alerts(critical_alerts[-3:])  # Show last 3 critical
    
    # Print position-level P&L
    final_prices = prices_test.iloc[-1]
    position_pnl = risk_manager.get_position_pnl(
        current_prices=final_prices,
        entry_prices=entry_prices,
        weights=final_weights
    )
    
    if len(position_pnl) > 0:
        print(f"\n📊 Position P&L Analysis:")
        print(f"   {'Ticker':<8} {'Weight':>8} {'Return':>10} {'Contribution':>12}")
        print(f"   {'-'*42}")
        for _, row in position_pnl.iterrows():
            print(f"   {row['ticker']:<8} {row['weight']:>+8.1%} {row['return']:>+10.1%} {row['contribution']:>+12.2%}")
    
    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    
    print("\n" + "=" * 70)
    print("📋 FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Best Strategy: {best_strategy_name}")
    
    print(f"\n📈 Backtest Performance:")
    print(f"   Total Return:   {best_strategy_results['total_return']:+.2%}")
    print(f"   Annual Return:  {best_strategy_results['annual_return']:+.2%}")
    print(f"   Sharpe Ratio:   {best_strategy_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:   {best_strategy_results['max_drawdown']:.2%}")
    
    print(f"\n🛡️ Risk Status:")
    print(f"   Current Drawdown: {risk_summary['current_drawdown']:.2%}")
    print(f"   Max Drawdown:     {risk_summary['max_drawdown']:.2%}")
    print(f"   VaR (95%):        {risk_summary['var_95']:.2%}")
    print(f"   Status:           {'🛑 STOPPED' if risk_summary['is_stopped_out'] else '✅ ACTIVE'}")
    
    print(f"\n📊 Current Portfolio (as of {latest_date.date()}):")
    print(f"   {'Ticker':<8} {'Weight':>10} {'Position':<8}")
    print(f"   {'-'*30}")
    
    for ticker, weight in final_weights.sort_values(ascending=False).items():
        if abs(weight) > 0.001:
            position = "LONG" if weight > 0 else "SHORT"
            print(f"   {ticker:<8} {weight:>+10.1%} {position:<8}")
    
    print(f"\n{'='*70}")
    print("✅ PORTFOLIO CONSTRUCTION COMPLETE!")
    print(f"{'='*70}\n")




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
