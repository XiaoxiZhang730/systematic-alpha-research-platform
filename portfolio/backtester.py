"""
Portfolio backtesting module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .optimizer import PortfolioOptimizer, PortfolioConstraints


class Backtester:
    """
    Backtest portfolio strategies.
    """
    
    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        rebalance_freq: str = 'M',  # Monthly
        transaction_cost: float = 0.001  # 10 bps
    ):
        """
        Initialize backtester.
        
        Args:
            optimizer: Portfolio optimizer instance
            rebalance_freq: Rebalance frequency ('D', 'W', 'M')
            transaction_cost: Transaction cost per trade (fraction)
        """
        self.optimizer = optimizer
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
    
    def run(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        method: str = 'signal_weighted'
    ) -> Dict:
        """
        Run backtest.
        
        Args:
            predictions: DataFrame of predictions (date x ticker)
            returns: DataFrame of returns (date x ticker)
            method: Optimization method
        
        Returns:
            Dictionary with backtest results
        """
        
        # Get rebalance dates
        if self.rebalance_freq == 'M':
            rebalance_dates = predictions.resample('M').last().index
        elif self.rebalance_freq == 'W':
            rebalance_dates = predictions.resample('W').last().index
        else:
            rebalance_dates = predictions.index
        
        # Initialize
        portfolio_values = [1.0]
        portfolio_returns = []
        weights_history = []
        turnover_history = []
        
        current_weights = pd.Series(0, index=predictions.columns)
        
        for i, date in enumerate(rebalance_dates[:-1]):
            next_date = rebalance_dates[i + 1]
            
            # Get predictions for this date
            if date not in predictions.index:
                continue
            
            pred = predictions.loc[date]
            
            # Optimize
            new_weights = self.optimizer.optimize(pred, returns, method=method)
            
            # Calculate turnover
            turnover = (new_weights - current_weights).abs().sum() / 2
            turnover_history.append(turnover)
            
            # Transaction costs
            tc = turnover * self.transaction_cost
            
            # Get returns between rebalance dates
            period_returns = returns.loc[date:next_date]
            
            if len(period_returns) > 0:
                # Portfolio return for the period
                daily_port_returns = (period_returns * new_weights).sum(axis=1)
                period_return = (1 + daily_port_returns).prod() - 1 - tc
                
                portfolio_returns.append(period_return)
                portfolio_values.append(portfolio_values[-1] * (1 + period_return))
            
            # Update weights
            current_weights = new_weights
            weights_history.append({
                'date': date,
                'weights': new_weights.to_dict()
            })
        
        # Calculate statistics
        returns_series = pd.Series(portfolio_returns)
        
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        annual_return = (1 + total_return) ** (12 / len(returns_series)) - 1 if len(returns_series) > 0 else 0
        volatility = returns_series.std() * np.sqrt(12) if len(returns_series) > 1 else 0
        sharpe = annual_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'avg_turnover': np.mean(turnover_history) if turnover_history else 0,
            'portfolio_values': portfolio_values,
            'returns': returns_series.tolist(),
            'weights_history': weights_history,
            'n_periods': len(returns_series)
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = values[0]
        max_dd = 0
        
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd


def print_backtest_results(results: Dict):
    """Print formatted backtest results."""
    
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n📈 Performance:")
    print(f"   Total Return:    {results['total_return']:+.2%}")
    print(f"   Annual Return:   {results['annual_return']:+.2%}")
    print(f"   Volatility:      {results['volatility']:.2%}")
    print(f"   Sharpe Ratio:    {results['sharpe']:.2f}")
    print(f"   Max Drawdown:    {results['max_drawdown']:.2%}")
    
    print(f"\n📊 Trading:")
    print(f"   Avg Turnover:    {results['avg_turnover']:.2%}")
    print(f"   # Periods:       {results['n_periods']}")
    
    print("=" * 60)
