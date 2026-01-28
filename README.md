# Systematic Portfolio Construction

A complete end-to-end quantitative portfolio construction framework that demonstrates the full workflow from raw data to portfolio optimization and risk management.

## Overview

This project implements a systematic approach to equity portfolio construction using machine learning for alpha generation. It covers the entire quantitative investment pipeline:

```
Data → Features → Model Training → Signal Generation → Portfolio Optimization → Backtesting → Risk Management
```

## Project Structure

```
systematic_portfolio_construction/
├── main.py                 # Main execution pipeline
├── configs/
│   └── config.py           # Configuration dataclass
├── data/
│   ├── load.py             # Data loading (Yahoo Finance)
│   └── preprocess.py       # Data cleaning
├── features/
│   ├── feature_engineering.py  # Feature creation & normalization
│   └── selection.py        # Feature selection utilities
├── models/
│   ├── ridge_model.py      # Ridge regression
│   ├── lasso_model.py      # Lasso regression
│   ├── random_forest_model.py  # Random Forest
│   ├── gbm_model.py        # Gradient Boosting
│   └── registry.py         # Model registry
├── tuning/
│   └── validator.py        # Hyperparameter tuning with time-series CV
├── execution/
│   └── multi_model_runner.py   # Multi-model comparison
├── evaluation/
│   ├── metrics.py          # IC, Rank IC, etc.
│   └── validation.py       # Quintile analysis
├── portfolio/
│   ├── optimizer.py        # Portfolio optimization methods
│   └── backtester.py       # Historical backtesting
├── risk/
│   └── post_trade.py       # Post-trade risk monitoring
└── runs/                   # Experiment logs (JSON)
```

## High-Level Design

### 1. Feature Engineering

19 alpha factors across multiple categories:

| Category | Features | Description |
|----------|----------|-------------|
| **Momentum** | `ret_5d`, `ret_10d`, `ret_21d`, `ret_63d`, `mom_12m_1m` | Multi-horizon return signals |
| **Volatility** | `vol_21d`, `vol_63d`, `vol_ratio`, `vol_skew` | Realized volatility metrics |
| **Mean Reversion** | `dist_ma_10d`, `dist_ma_21d`, `dist_ma_50d` | Distance from moving averages |
| **Technical** | `rsi_norm`, `macd_norm`, `bb_position`, `atr_norm` | Classic technical indicators |
| **Volume** | `volume_ratio`, `log_dollar_vol`, `pv_corr_21d` | Volume-based signals |

**Key Design Choices:**
- Cross-sectional z-score normalization (not time-series)
- Winsorization at 1st/99th percentiles
- 1-day lag on all features to prevent look-ahead bias
- Target: 21-day forward return, cross-sectionally demeaned

### 2. Model Training

Four models compared with time-series cross-validation:

- **Ridge Regression**: L2 regularization, baseline linear model
- **Lasso Regression**: L1 regularization, sparse feature selection
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting (GBM)**: Sequential boosting

**Evaluation Metric:** Rank IC (Spearman correlation between predictions and realized returns)

### 3. Portfolio Optimization

Three weighting schemes implemented:

| Method | Description |
|--------|-------------|
| **Signal Weighted** | Weights proportional to model predictions |
| **Rank Weighted** | Weights based on prediction ranks |
| **Long-Short** | Top N long, bottom N short (equal weight within) |

**Constraints:**
- Max single position: 15%
- Max short exposure: 30%
- Max leverage: 1.3x

### 4. Backtesting

- Monthly rebalancing
- 10 bps transaction cost
- Performance metrics: Sharpe, Sortino, Calmar, Max Drawdown

### 5. Risk Management

Post-trade monitoring with:
- Drawdown limits (15%)
- Position stop-loss (10%)
- VaR monitoring (95% confidence)
- Daily risk reports

## Results Summary

### Model Comparison

| Model | R² | Rank IC | IC Std | Hit Rate | L/S Spread |
|-------|-----|---------|--------|----------|------------|
| Ridge | 0.0027 | 0.0249 | 0.3139 | 53.76% | 0.0112 |
| Lasso | 0.0057 | 0.0251 | 0.3325 | 53.35% | 0.0112 |
| **Random Forest** | **0.0085** | **0.0813** | 0.2891 | **61.15%** | **0.0211** |
| GBM | -0.0493 | 0.0355 | 0.2847 | 55.68% | 0.0066 |

**Best Model:** Random Forest
- Rank IC: 0.0813 (strong predictive signal)
- IC IR: 0.2813 (consistent performance)
- Monotonic quintile returns: ✅ Yes

### Backtest Performance

| Strategy | Total Return | Annual Return | Sharpe | Max DD |
|----------|-------------|---------------|--------|--------|
| Signal Weighted | +28.16% | +9.15% | 0.68 | 16.44% |
| Rank Weighted | +34.32% | +10.98% | 0.73 | 16.63% |
| **Long-Short (5/5)** | **+68.76%** | **+20.29%** | **0.93** | 23.98% |
| Equal Weight (baseline) | +55.77% | +16.93% | 0.61 | 32.35% |

**Test Period:** Jan 2022 - Dec 2024

**Quintile Analysis (Random Forest):**
```
Q1 (lowest pred): -1.09%
Q2:               -0.38%
Q3:               -0.03%
Q4:               +0.46%
Q5 (highest pred): +1.03%
```
✅ Monotonic relationship confirmed

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/systematic-portfolio-construction.git
cd systematic-portfolio-construction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.0
scipy>=1.11.0
```

## Usage

### Quick Start

```bash
# Run the full pipeline
python main.py
```

This will execute all 6 phases:
1. Hyperparameter tuning
2. Model training & evaluation
3. Portfolio optimization
4. Backtesting
5. Strategy comparison
6. Risk monitoring

### Configuration

Edit `configs/config.py` or modify directly in `main.py`:

```python
cfg = Config(
    tickers=("AAPL", "MSFT", "GOOGL", ...),  # Stock universe
    market="SPY",                             # Benchmark
    start="2015-01-01",                       # Data start
    end="2024-12-31",                         # Data end
    split_date=pd.Timestamp("2021-12-31")     # Train/test split
)
```

### Running Individual Components

```python
# Just tune models
from tuning.validator import tune_all_models
results = tune_all_models(X_train, y_train, cfg)

# Just run backtest
from portfolio import Backtester
backtester = Backtester(returns, predictions)
results = backtester.run(method='long_short')
```

## Methodology Notes

### Feature Normalization
Cross-sectional z-score normalization is used rather than time-series normalization. This aligns with the goal of cross-sectional stock selection—identifying which stocks will outperform peers on a given day, rather than predicting absolute price movements.

### Target Construction
The target variable is the 21-day forward log return, cross-sectionally demeaned. This design choice:
- Focuses the model on relative performance prediction
- Removes market beta exposure from the target
- Aligns with long-short portfolio construction

### Evaluation Metric
Rank IC (Spearman correlation) is used as the primary evaluation metric rather than R² or MSE. This reflects the practical goal of ranking stocks correctly for portfolio construction, where the magnitude of predictions matters less than their relative ordering.

## Known Limitations

1. **Survivorship Bias**: Uses current stock universe; stocks that delisted/failed are not included
2. **Sector Concentration**: Limited to Tech & Finance sectors (18 stocks in base config)
3. **Transaction Costs**: Simplified model (flat 10 bps)
4. **No Market Impact**: Assumes unlimited liquidity
5. **Look-Ahead in Universe**: Tickers selected with knowledge of future success

## Future Improvements

- [ ] Add sector-neutral constraints
- [ ] Implement idiosyncratic volatility (beta-adjusted)
- [ ] Add fundamental factors (value, quality)
- [ ] Expand universe to 100+ stocks
- [ ] Add regime detection
- [ ] Implement transaction cost optimization

## License

MIT License

## Acknowledgments

- Data provided by Yahoo Finance via `yfinance`
- Inspired by quantitative finance literature and industry practices
