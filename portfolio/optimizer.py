"""
Portfolio optimization module.
Supports multiple optimization methods with risk constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioConstraints:
    """Portfolio constraints configuration."""
    max_weight: float = 0.10          # Max weight per stock (10%)
    min_weight: float = -0.10         # Min weight (negative = short)
    max_long_weight: float = 1.0      # Total long exposure
    max_short_weight: float = 0.0     # Total short exposure (0 = long only)
    max_turnover: float = 1.0         # Max turnover per rebalance
    max_leverage: float = 1.0         # Max leverage (1.0 = no leverage)


class PortfolioOptimizer:
    """
    Portfolio optimizer supporting multiple methods.
    
    Methods:
        - signal_weighted: Weight proportional to predictions
        - mean_variance: Classic Markowitz optimization
        - risk_parity: Equal risk contribution
        - min_variance: Minimum variance portfolio
        - max_sharpe: Maximum Sharpe ratio
        - equal_weight: Simple equal weighting
    """
    
    def __init__(
        self,
        constraints: Optional[PortfolioConstraints] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize optimizer.
        
        Args:
            constraints: Portfolio constraints
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.constraints = constraints or PortfolioConstraints()
        self.risk_free_rate = risk_free_rate
    
    def optimize(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        method: str = "signal_weighted",
        **kwargs
    ) -> pd.Series:
        """
        Optimize portfolio weights.
        
        Args:
            predictions: Model predictions (higher = more bullish)
            returns: Historical returns for risk estimation (optional)
            method: Optimization method
            **kwargs: Additional method-specific arguments
        
        Returns:
            Series of portfolio weights (indexed by ticker)
        """
        
        method_map = {
            'signal_weighted': self._signal_weighted,
            'rank_weighted': self._rank_weighted,
            'mean_variance': self._mean_variance,
            'min_variance': self._min_variance,
            'max_sharpe': self._max_sharpe,
            'risk_parity': self._risk_parity,
            'equal_weight': self._equal_weight,
            'long_short': self._long_short,
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")
        
        weights = method_map[method](predictions, returns, **kwargs)
        weights = self._apply_constraints(weights)
        
        return weights
    
    # ================================================================
    # OPTIMIZATION METHODS
    # ================================================================
    
    def _signal_weighted(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.Series:
        """
        Weight stocks proportionally to their predicted returns.
        
        Formula: w_i = pred_i / sum(|pred|)
        """
        # Demean predictions (go long high, short low)
        pred_demeaned = predictions - predictions.mean()
        
        # Normalize to sum of absolute weights = 1
        abs_sum = np.abs(pred_demeaned).sum()
        if abs_sum > 0:
            weights = pred_demeaned / abs_sum
        else:
            weights = pd.Series(0, index=predictions.index)
        
        return weights
    
    def _rank_weighted(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.Series:
        """
        Weight stocks based on prediction ranks.
        More robust to outliers than signal_weighted.
        
        Formula: w_i = (rank_i - median_rank) / sum(|rank - median|)
        """
        # Convert to ranks (0 to 1)
        ranks = predictions.rank(pct=True)
        
        # Center around 0.5 (median)
        centered = ranks - 0.5
        
        # Normalize
        abs_sum = np.abs(centered).sum()
        if abs_sum > 0:
            weights = centered / abs_sum
        else:
            weights = pd.Series(0, index=predictions.index)
        
        return weights
    
    def _long_short(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        n_long: int = 5,
        n_short: int = 5,
        **kwargs
    ) -> pd.Series:
        """
        Long top N stocks, short bottom N stocks (equal weight within each).
        
        Classic long-short portfolio based on signal quintiles.
        """
        n_stocks = len(predictions)
        weights = pd.Series(0.0, index=predictions.index)
        
        # Sort by predictions
        sorted_pred = predictions.sort_values(ascending=False)
        
        # Long top N
        long_stocks = sorted_pred.head(n_long).index
        weights[long_stocks] = 1.0 / n_long
        
        # Short bottom N
        short_stocks = sorted_pred.tail(n_short).index
        weights[short_stocks] = -1.0 / n_short
        
        return weights
    
    def _equal_weight(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.Series:
        """
        Equal weight all stocks (baseline).
        """
        n = len(predictions)
        weights = pd.Series(1.0 / n, index=predictions.index)
        return weights
    
    def _mean_variance(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        risk_aversion: float = 1.0,
        **kwargs
    ) -> pd.Series:
        """
        Classic Markowitz mean-variance optimization.
        
        Maximize: w'μ - (λ/2) * w'Σw
        Subject to: sum(w) = 1, w_min <= w <= w_max
        """
        if returns is None:
            # Fall back to signal weighted if no returns provided
            return self._signal_weighted(predictions, returns)
        
        n = len(predictions)
        tickers = predictions.index
        
        # Expected returns (use predictions as proxy)
        mu = predictions.values
        
        # Covariance matrix
        cov = returns[tickers].cov().values
        
        # Objective: minimize negative utility (= maximize utility)
        def objective(w):
            port_return = w @ mu
            port_var = w @ cov @ w
            utility = port_return - (risk_aversion / 2) * port_var
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=tickers)
        else:
            # Fall back to equal weight
            weights = pd.Series(1.0 / n, index=tickers)
        
        return weights
    
    def _min_variance(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.Series:
        """
        Minimum variance portfolio.
        
        Ignores expected returns, minimizes portfolio variance.
        """
        if returns is None:
            return self._equal_weight(predictions, returns)
        
        n = len(predictions)
        tickers = predictions.index
        
        # Covariance matrix
        cov = returns[tickers].cov().values
        
        # Objective: minimize variance
        def objective(w):
            return w @ cov @ w
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(0, self.constraints.max_weight)] * n  # Long only for min var
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=tickers)
        else:
            weights = pd.Series(1.0 / n, index=tickers)
        
        return weights
    
    def _max_sharpe(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.Series:
        """
        Maximum Sharpe ratio portfolio.
        """
        if returns is None:
            return self._signal_weighted(predictions, returns)
        
        n = len(predictions)
        tickers = predictions.index
        
        # Expected returns (use predictions)
        mu = predictions.values
        
        # Covariance matrix
        cov = returns[tickers].cov().values
        
        # Risk-free rate (daily)
        rf_daily = self.risk_free_rate / 252
        
        # Objective: minimize negative Sharpe ratio
        def objective(w):
            port_return = w @ mu
            port_std = np.sqrt(w @ cov @ w)
            if port_std < 1e-10:
                return 1e10
            sharpe = (port_return - rf_daily) / port_std
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(0, self.constraints.max_weight)] * n
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=tickers)
        else:
            weights = pd.Series(1.0 / n, index=tickers)
        
        return weights
    
    def _risk_parity(
        self,
        predictions: pd.Series,
        returns: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.Series:
        """
        Risk parity: Equal risk contribution from each asset.
        """
        if returns is None:
            return self._equal_weight(predictions, returns)
        
        n = len(predictions)
        tickers = predictions.index
        
        # Covariance matrix
        cov = returns[tickers].cov().values
        
        # Objective: minimize sum of squared differences in risk contribution
        def objective(w):
            port_var = w @ cov @ w
            if port_var < 1e-10:
                return 1e10
            
            # Marginal risk contribution
            mrc = cov @ w
            # Risk contribution
            rc = w * mrc / np.sqrt(port_var)
            # Target: equal risk contribution
            target_rc = np.sqrt(port_var) / n
            # Sum of squared differences
            return np.sum((rc - target_rc) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds (long only for risk parity)
        bounds = [(0.01, self.constraints.max_weight)] * n
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=tickers)
        else:
            weights = pd.Series(1.0 / n, index=tickers)
        
        return weights
    
    # ================================================================
    # CONSTRAINTS
    # ================================================================
    
    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply portfolio constraints to weights."""
        
        w = weights.copy()
        
        # Clip individual weights
        w = w.clip(
            lower=self.constraints.min_weight,
            upper=self.constraints.max_weight
        )
        
        # Handle leverage constraint
        long_sum = w[w > 0].sum()
        short_sum = abs(w[w < 0].sum())
        
        # Scale down if exceeds leverage
        total_exposure = long_sum + short_sum
        if total_exposure > self.constraints.max_leverage:
            w = w * (self.constraints.max_leverage / total_exposure)
        
        # Enforce long/short limits
        if long_sum > self.constraints.max_long_weight:
            w[w > 0] = w[w > 0] * (self.constraints.max_long_weight / long_sum)
        
        if short_sum > self.constraints.max_short_weight:
            if self.constraints.max_short_weight == 0:
                w[w < 0] = 0
            else:
                w[w < 0] = w[w < 0] * (self.constraints.max_short_weight / short_sum)
        
        # Renormalize if needed (for long-only)
        if self.constraints.min_weight >= 0:
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
        
        return w
    
    # ================================================================
    # ANALYSIS
    # ================================================================
    
    def analyze_portfolio(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        predictions: pd.Series
    ) -> Dict:
        """
        Analyze portfolio characteristics.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns
            predictions: Model predictions
        
        Returns:
            Dictionary with portfolio statistics
        """
        tickers = weights.index
        
        # Basic stats
        n_long = (weights > 0.001).sum()
        n_short = (weights < -0.001).sum()
        long_exposure = weights[weights > 0].sum()
        short_exposure = abs(weights[weights < 0].sum())
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        
        # Top holdings
        top_long = weights.nlargest(5)
        top_short = weights.nsmallest(5)
        
        # Portfolio expected return (based on predictions)
        expected_return = (weights * predictions).sum()
        
        # Portfolio risk (if returns available)
        if returns is not None and len(returns) > 0:
            port_returns = (returns[tickers] * weights).sum(axis=1)
            volatility = port_returns.std() * np.sqrt(252)
            sharpe = expected_return / volatility if volatility > 0 else 0
        else:
            volatility = None
            sharpe = None
        
        return {
            'n_stocks': len(weights),
            'n_long': n_long,
            'n_short': n_short,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'top_long': top_long.to_dict(),
            'top_short': top_short.to_dict(),
            'max_weight': weights.max(),
            'min_weight': weights.min(),
        }


def print_portfolio_summary(weights: pd.Series, analysis: Dict):
    """Print formatted portfolio summary."""
    
    print("\n" + "=" * 60)
    print("📊 PORTFOLIO SUMMARY")
    print("=" * 60)
    
    print(f"\n📈 Exposure:")
    print(f"   Long:  {analysis['n_long']:3d} stocks, {analysis['long_exposure']:6.1%} exposure")
    print(f"   Short: {analysis['n_short']:3d} stocks, {analysis['short_exposure']:6.1%} exposure")
    print(f"   Net:   {analysis['net_exposure']:6.1%}")
    print(f"   Gross: {analysis['gross_exposure']:6.1%}")
    
    print(f"\n📊 Risk/Return:")
    print(f"   Expected Return: {analysis['expected_return']:+.4f}")
    if analysis['volatility']:
        print(f"   Volatility:      {analysis['volatility']:.2%}")
    if analysis['sharpe']:
        print(f"   Sharpe Ratio:    {analysis['sharpe']:.2f}")
    
    print(f"\n🔝 Top Long Positions:")
    for ticker, weight in analysis['top_long'].items():
        if weight > 0.001:
            print(f"   {ticker:<6} {weight:6.1%}")
    
    if analysis['n_short'] > 0:
        print(f"\n🔻 Top Short Positions:")
        for ticker, weight in analysis['top_short'].items():
            if weight < -0.001:
                print(f"   {ticker:<6} {weight:6.1%}")
    
    print(f"\n📋 Weight Range: [{analysis['min_weight']:.1%}, {analysis['max_weight']:.1%}]")
    print("=" * 60)
