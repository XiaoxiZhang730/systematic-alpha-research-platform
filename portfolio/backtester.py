"""
Portfolio Backtesting Module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .optimizer import PortfolioOptimizer, PortfolioConstraints


class Backtester:
    """Backtest portfolio strategies on historical data."""
    
    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        rebalance_freq: str = 'M',
        transaction_cost: float = 0.001,
        slippage: float = 0.0005
    ):
        self.optimizer = optimizer
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        self.portfolio_values = []
        self.portfolio_returns = []
        self.weights_history = []
        self.turnover_history = []
        self.dates = []
    
    def run(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        method: str = 'signal_weighted',
        **method_kwargs
    ) -> Dict:
        """Run backtest simulation."""
        
        print(f"\n{'─'*60}")
        print(f"🔄 Running Backtest: {method.upper()}")
        print(f"{'─'*60}")
        print(f"   Rebalance frequency: {self._freq_to_name(self.rebalance_freq)}")
        print(f"   Transaction cost: {self.transaction_cost:.2%}")
        
        # Reset state
        self._reset()
        
        # Get unique dates from predictions
        if isinstance(predictions.index, pd.MultiIndex):
            all_dates = predictions.index.get_level_values('date').unique().sort_values()
        else:
            all_dates = predictions.index.unique().sort_values()
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(all_dates)
        
        print(f"   Backtest period: {rebalance_dates[0].date()} to {rebalance_dates[-1].date()}")
        print(f"   Number of rebalances: {len(rebalance_dates) - 1}")
        
        # Initialize
        portfolio_value = 1.0
        current_weights = None
        
        # Main backtest loop
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]
            
            # Get predictions for current date
            try:
                if isinstance(predictions.index, pd.MultiIndex):
                    current_pred = predictions.xs(current_date, level='date')
                else:
                    current_pred = predictions.loc[current_date]
            except KeyError:
                continue
            
            if len(current_pred) == 0:
                continue
            
            # Optimize portfolio
            try:
                new_weights = self.optimizer.optimize(
                    predictions=current_pred,
                    returns=returns,
                    method=method,
                    **method_kwargs
                )
            except Exception as e:
                print(f"   Warning: Optimization failed for {current_date.date()}: {e}")
                continue
            
            # Calculate turnover
            if current_weights is not None:
                all_tickers = new_weights.index.union(current_weights.index)
                new_w = new_weights.reindex(all_tickers, fill_value=0)
                old_w = current_weights.reindex(all_tickers, fill_value=0)
                turnover = (new_w - old_w).abs().sum() / 2
            else:
                turnover = new_weights.abs().sum() / 2
            
            self.turnover_history.append(turnover)
            
            # Transaction costs
            total_cost = turnover * (self.transaction_cost + self.slippage)
            
            # Get returns for the period
            try:
                period_returns = returns.loc[current_date:next_date].iloc[1:]
            except:
                continue
            
            if len(period_returns) == 0:
                continue
            
            # Get available tickers
            portfolio_tickers = new_weights.index.tolist()
            available_tickers = [t for t in portfolio_tickers if t in period_returns.columns]
            
            if len(available_tickers) == 0:
                continue
            
            # Align weights
            weights_aligned = new_weights[available_tickers]
            if weights_aligned.abs().sum() > 0:
                weights_aligned = weights_aligned / weights_aligned.abs().sum() * new_weights.abs().sum()
            
            # Calculate portfolio returns
            daily_port_returns = (period_returns[available_tickers] * weights_aligned).sum(axis=1)
            period_return = (1 + daily_port_returns).prod() - 1
            period_return_net = period_return - total_cost
            
            # Update portfolio value
            portfolio_value = portfolio_value * (1 + period_return_net)
            
            # Store results
            self.portfolio_values.append(portfolio_value)
            self.portfolio_returns.append(period_return_net)
            self.dates.append(next_date)
            self.weights_history.append({
                'date': current_date,
                'weights': new_weights.to_dict(),
                'turnover': turnover,
                'cost': total_cost
            })
            
            current_weights = new_weights
        
        # Calculate metrics
        results = self._calculate_metrics()
        self._print_summary(results)
        
        return results
    
    def _reset(self):
        """Reset backtest state."""
        self.portfolio_values = [1.0]
        self.portfolio_returns = []
        self.weights_history = []
        self.turnover_history = []
        self.dates = [None]
    
    def _get_rebalance_dates(self, all_dates):
        """Get rebalance dates based on frequency."""
        dates_series = pd.Series(all_dates, index=all_dates)
        
        if self.rebalance_freq == 'D':
            return all_dates
        elif self.rebalance_freq == 'W':
            return pd.DatetimeIndex(dates_series.resample('W').last().dropna().values)
        elif self.rebalance_freq == 'M':
            return pd.DatetimeIndex(dates_series.resample('M').last().dropna().values)
        else:
            return all_dates
    
    def _freq_to_name(self, freq):
        """Convert frequency code to name."""
        return {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}.get(freq, freq)
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        
        if len(self.portfolio_returns) < 2:
            return {
                'total_return': 0, 'annual_return': 0, 'volatility': 0,
                'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0,
                'calmar_ratio': 0, 'win_rate': 0, 'avg_turnover': 0,
                'total_costs': 0, 'n_periods': 0,
                'portfolio_values': self.portfolio_values,
                'returns': self.portfolio_returns, 'dates': self.dates
            }
        
        returns_series = pd.Series(self.portfolio_returns)
        
        # Returns
        total_return = self.portfolio_values[-1] / self.portfolio_values[0] - 1
        periods_per_year = {'D': 252, 'W': 52, 'M': 12}.get(self.rebalance_freq, 12)
        n_years = len(returns_series) / periods_per_year
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Risk
        volatility = returns_series.std() * np.sqrt(periods_per_year)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Sortino
        neg_returns = returns_series[returns_series < 0]
        downside_vol = neg_returns.std() * np.sqrt(periods_per_year) if len(neg_returns) > 0 else 0
        sortino_ratio = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Other metrics
        win_rate = (returns_series > 0).sum() / len(returns_series)
        avg_turnover = np.mean(self.turnover_history) if self.turnover_history else 0
        total_costs = sum(w.get('cost', 0) for w in self.weights_history)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'total_costs': total_costs,
            'n_periods': len(returns_series),
            'portfolio_values': self.portfolio_values,
            'returns': self.portfolio_returns,
            'dates': self.dates,
            'weights_history': self.weights_history
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.portfolio_values) < 2:
            return 0
        
        values = np.array(self.portfolio_values)
        peak = values[0]
        max_dd = 0
        
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _print_summary(self, results: Dict):
        """Print backtest summary."""
        
        print(f"\n{'='*60}")
        print("📊 BACKTEST RESULTS")
        print(f"{'='*60}")
        
        print(f"\n📈 PERFORMANCE")
        print(f"   Total Return:     {results['total_return']:+.2%}")
        print(f"   Annual Return:    {results['annual_return']:+.2%}")
        print(f"   Volatility:       {results['volatility']:.2%}")
        
        print(f"\n⚖️  RISK-ADJUSTED")
        print(f"   Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio:    {results['sortino_ratio']:.2f}")
        print(f"   Calmar Ratio:     {results['calmar_ratio']:.2f}")
        
        print(f"\n📉 RISK")
        print(f"   Max Drawdown:     {results['max_drawdown']:.2%}")
        print(f"   Win Rate:         {results['win_rate']:.1%}")
        
        print(f"\n💰 TRADING")
        print(f"   Avg Turnover:     {results['avg_turnover']:.1%} per period")
        print(f"   Total Costs:      {results['total_costs']:.2%}")
        print(f"   # Rebalances:     {results['n_periods']}")
        
        print(f"{'='*60}")


def print_backtest_comparison(results_dict: Dict[str, Dict]):
    """Print comparison of multiple backtest results."""
    
    print(f"\n{'='*90}")
    print("📊 BACKTEST COMPARISON")
    print(f"{'='*90}")
    
    print(f"\n{'Method':<20} {'Total':>10} {'Annual':>10} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Turnover':>10}")
    print("-" * 90)
    
    for method, results in results_dict.items():
        total = results['total_return']
        annual = results['annual_return']
        vol = results['volatility']
        sharpe = results['sharpe_ratio']
        max_dd = results['max_drawdown']
        turnover = results['avg_turnover']
        
        print(f"{method:<20} {total:>+10.2%} {annual:>+10.2%} {vol:>8.2%} {sharpe:>8.2f} {max_dd:>8.2%} {turnover:>10.1%}")
    
    print("-" * 90)
    
    # Find best by Sharpe ratio
    best_method = max(results_dict.items(), key=lambda x: x[1]['sharpe_ratio'])
    print(f"\n🏆 Best Strategy (by Sharpe): {best_method[0]}")
    print(f"   Sharpe Ratio: {best_method[1]['sharpe_ratio']:.2f}")
    print(f"   Annual Return: {best_method[1]['annual_return']:+.2%}")
    
    print(f"{'='*90}")
