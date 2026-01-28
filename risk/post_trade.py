"""
Post-Trade Risk Management Module.

Monitors portfolio after positions are taken:
- Drawdown monitoring
- P&L tracking
- Stop-loss checking
- Risk alerts
- Performance reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAlert:
    """Risk alert container."""
    timestamp: datetime
    level: AlertLevel
    category: str
    message: str
    value: float = 0.0
    threshold: float = 0.0


@dataclass
class DailyRiskReport:
    """Daily risk report container."""
    date: datetime
    
    # P&L
    daily_pnl: float = 0.0
    daily_return: float = 0.0
    cumulative_pnl: float = 0.0
    cumulative_return: float = 0.0
    
    # Drawdown
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    days_in_drawdown: int = 0
    
    # Volatility
    realized_volatility: float = 0.0
    
    # VaR
    var_95: float = 0.0
    var_99: float = 0.0
    
    # Exposure
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    
    # Alerts
    alerts: List[RiskAlert] = field(default_factory=list)


class PostTradeRiskManager:
    """
    Post-trade risk management system.
    
    Monitors:
    - Drawdown
    - P&L
    - Stop-loss levels
    - Risk metrics
    - Generates alerts
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        position_stop_loss: float = 0.10,
        portfolio_stop_loss: float = 0.05,
        var_limit_95: float = 0.03,
        volatility_lookback: int = 21,
        initial_value: Optional[float] = None 
    ):
        """
        Initialize post-trade risk manager.
        
        Args:
            max_drawdown: Maximum allowed drawdown (e.g., 0.15 = 15%)
            position_stop_loss: Stop-loss per position (e.g., 0.10 = 10%)
            portfolio_stop_loss: Daily portfolio stop-loss (e.g., 0.05 = 5%)
            var_limit_95: Maximum 95% VaR allowed (e.g., 0.03 = 3%)
            volatility_lookback: Days for volatility calculation
        """
        self.max_drawdown = max_drawdown
        self.position_stop_loss = position_stop_loss
        self.portfolio_stop_loss = portfolio_stop_loss
        self.var_limit_95 = var_limit_95
        self.volatility_lookback = volatility_lookback
        
        # # State tracking
        # self.portfolio_values = []
        # self.portfolio_returns = []
        # self.peak_value = 1.0
        # self.current_value = 1.0
        # self.start_value = 1.0

        # State tracking - CHANGED: Use initial_value instead of hardcoded 1.0
        self.portfolio_values = []
        self.portfolio_returns = []
        self.peak_value = initial_value if initial_value else 0.0      # <-- CHANGED
        self.current_value = initial_value if initial_value else 0.0   # <-- CHANGED
        self.start_value = initial_value if initial_value else 0.0     # <-- CHANGED
        self._initialized = initial_value is not None 
        
        # Drawdown tracking
        self.drawdown_history = []
        self.drawdown_start_date = None
        self.days_in_drawdown = 0
        
        # Alerts
        self.alerts = []
        self.is_stopped_out = False
    
    def update(
        self,
        date: datetime,
        portfolio_value: float,
        weights: pd.Series,
        prices: Optional[pd.Series] = None,
        entry_prices: Optional[pd.Series] = None
    ) -> DailyRiskReport:
        """
        Daily update of risk monitoring.
        
        Args:
            date: Current date
            portfolio_value: Current portfolio value
            weights: Current portfolio weights
            prices: Current prices (optional)
            entry_prices: Entry prices for positions (optional)
        
        Returns:
            DailyRiskReport with all metrics and alerts
        """
        daily_alerts = []
        
        if not self._initialized:
            self.start_value = portfolio_value
            self.peak_value = portfolio_value
            self.current_value = portfolio_value
            self._initialized = True
            
        # Calculate daily return
        if len(self.portfolio_values) > 0:
            prev_value = self.portfolio_values[-1]
            daily_return = (portfolio_value - prev_value) / prev_value
            daily_pnl = portfolio_value - prev_value
        else:
            daily_return = 0.0
            daily_pnl = 0.0
        
        # Update history
        self.portfolio_values.append(portfolio_value)
        self.portfolio_returns.append(daily_return)
        self.current_value = portfolio_value
        
        # Cumulative P&L
        cumulative_return = (portfolio_value - self.start_value) / self.start_value
        cumulative_pnl = portfolio_value - self.start_value
        
        # ===== DRAWDOWN MONITORING =====
        current_drawdown, max_drawdown = self._update_drawdown(portfolio_value, date)
        
        # Alert: Drawdown warning (50% of limit)
        if current_drawdown > self.max_drawdown * 0.5 and current_drawdown <= self.max_drawdown:
            alert = RiskAlert(
                timestamp=date,
                level=AlertLevel.WARNING,
                category="DRAWDOWN",
                message=f"Drawdown at {current_drawdown:.1%}, approaching limit of {self.max_drawdown:.1%}",
                value=current_drawdown,
                threshold=self.max_drawdown
            )
            daily_alerts.append(alert)
        
        # Alert: Drawdown breach
        if current_drawdown > self.max_drawdown:
            alert = RiskAlert(
                timestamp=date,
                level=AlertLevel.CRITICAL,
                category="DRAWDOWN",
                message=f"DRAWDOWN LIMIT BREACHED: {current_drawdown:.1%} > {self.max_drawdown:.1%}",
                value=current_drawdown,
                threshold=self.max_drawdown
            )
            daily_alerts.append(alert)
        
        # ===== PORTFOLIO STOP-LOSS CHECK =====
        if daily_return < -self.portfolio_stop_loss:
            alert = RiskAlert(
                timestamp=date,
                level=AlertLevel.CRITICAL,
                category="STOP_LOSS",
                message=f"PORTFOLIO STOP-LOSS TRIGGERED: {daily_return:.1%} loss today",
                value=daily_return,
                threshold=-self.portfolio_stop_loss
            )
            daily_alerts.append(alert)
            self.is_stopped_out = True
        
        # ===== POSITION STOP-LOSS CHECK =====
        if prices is not None and entry_prices is not None:
            stopped_positions = self._check_position_stops(prices, entry_prices, weights)
            for ticker, loss in stopped_positions:
                alert = RiskAlert(
                    timestamp=date,
                    level=AlertLevel.WARNING,
                    category="POSITION_STOP",
                    message=f"Position stop-loss: {ticker} down {loss:.1%}",
                    value=loss,
                    threshold=self.position_stop_loss
                )
                daily_alerts.append(alert)
        
        # ===== VOLATILITY & VAR =====
        realized_vol, var_95, var_99 = self._calculate_risk_metrics()
        
        # Alert: VaR breach
        if var_95 > self.var_limit_95:
            alert = RiskAlert(
                timestamp=date,
                level=AlertLevel.WARNING,
                category="VAR",
                message=f"VaR(95%) at {var_95:.2%} exceeds limit of {self.var_limit_95:.2%}",
                value=var_95,
                threshold=self.var_limit_95
            )
            daily_alerts.append(alert)
        
        # ===== EXPOSURE =====
        net_exposure = weights.sum()
        gross_exposure = weights.abs().sum()
        
        # Store alerts
        self.alerts.extend(daily_alerts)
        
        # Build report
        report = DailyRiskReport(
            date=date,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            cumulative_pnl=cumulative_pnl,
            cumulative_return=cumulative_return,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            days_in_drawdown=self.days_in_drawdown,
            realized_volatility=realized_vol,
            var_95=var_95,
            var_99=var_99,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            alerts=daily_alerts
        )
        
        return report
    
    def _update_drawdown(self, portfolio_value: float, date: datetime) -> Tuple[float, float]:
        """Update drawdown tracking."""
        
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.drawdown_start_date = None
            self.days_in_drawdown = 0
        else:
            if self.drawdown_start_date is None:
                self.drawdown_start_date = date
            self.days_in_drawdown += 1
        
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.drawdown_history.append(current_drawdown)
        max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0
        
        return current_drawdown, max_drawdown
    
    def _check_position_stops(
        self,
        current_prices: pd.Series,
        entry_prices: pd.Series,
        weights: pd.Series
    ) -> List[Tuple[str, float]]:
        """Check if any positions have hit stop-loss."""
        
        stopped = []
        
        for ticker in weights.index:
            if ticker not in entry_prices or ticker not in current_prices:
                continue
            
            weight = weights[ticker]
            if abs(weight) < 0.001:
                continue
            
            entry = entry_prices[ticker]
            current = current_prices[ticker]
            
            if entry <= 0:
                continue
            
            position_return = (current - entry) / entry
            
            # Long position stop
            if weight > 0 and position_return < -self.position_stop_loss:
                stopped.append((ticker, position_return))
            
            # Short position stop
            if weight < 0 and position_return > self.position_stop_loss:
                stopped.append((ticker, -position_return))
        
        return stopped
    
    def _calculate_risk_metrics(self) -> Tuple[float, float, float]:
        """Calculate realized volatility and VaR."""
        
        if len(self.portfolio_returns) < 5:
            return 0.0, 0.0, 0.0
        
        returns = np.array(self.portfolio_returns[-self.volatility_lookback:])
        
        realized_vol = np.std(returns) * np.sqrt(252)
        var_95 = -np.percentile(returns, 5) if len(returns) >= 20 else 0
        var_99 = -np.percentile(returns, 1) if len(returns) >= 100 else 0
        
        return realized_vol, max(var_95, 0), max(var_99, 0)
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary."""
        
        realized_vol, var_95, var_99 = self._calculate_risk_metrics()
        
        current_dd = self.drawdown_history[-1] if self.drawdown_history else 0
        max_dd = max(self.drawdown_history) if self.drawdown_history else 0
        
        # Sharpe ratio
        if len(self.portfolio_returns) > 21:
            returns = np.array(self.portfolio_returns)
            mean_return = np.mean(returns) * 252
            vol = np.std(returns) * np.sqrt(252)
            sharpe = mean_return / vol if vol > 0 else 0
        else:
            sharpe = 0.0
        
        # Win rate
        if len(self.portfolio_returns) > 0:
            wins = sum(1 for r in self.portfolio_returns if r > 0)
            win_rate = wins / len(self.portfolio_returns)
        else:
            win_rate = 0.0
        
        return {
            'total_return': (self.current_value - self.start_value) / self.start_value,
            'realized_volatility': realized_vol,
            'sharpe_ratio': sharpe,
            'var_95': var_95,
            'var_99': var_99,
            'current_drawdown': current_dd,
            'max_drawdown': max_dd,
            'days_in_drawdown': self.days_in_drawdown,
            'win_rate': win_rate,
            'n_days': len(self.portfolio_returns),
            'is_stopped_out': self.is_stopped_out,
            'n_alerts': len(self.alerts)
        }
    
    def get_position_pnl(
        self,
        current_prices: pd.Series,
        entry_prices: pd.Series,
        weights: pd.Series
    ) -> pd.DataFrame:
        """Calculate P&L for each position."""
        
        data = []
        
        for ticker in weights.index:
            weight = weights[ticker]
            
            if abs(weight) < 0.001:
                continue
            
            entry = entry_prices.get(ticker, current_prices.get(ticker, 0))
            current = current_prices.get(ticker, 0)
            
            if entry > 0:
                pct_return = (current - entry) / entry
                if weight < 0:
                    pct_return = -pct_return
            else:
                pct_return = 0.0
            
            contribution = weight * pct_return
            
            data.append({
                'ticker': ticker,
                'weight': weight,
                'entry_price': entry,
                'current_price': current,
                'return': pct_return,
                'contribution': contribution
            })
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            df = df.sort_values('contribution', ascending=False)
        
        return df
    
    def get_alerts(self, level: Optional[AlertLevel] = None) -> List[RiskAlert]:
        """Get alerts, optionally filtered by level."""
        if level is None:
            return self.alerts
        return [a for a in self.alerts if a.level == level]
    
    def reset(self, start_value: float = 1.0):
        """Reset the risk manager state."""
        self.portfolio_values = []
        self.portfolio_returns = []
        self.peak_value = start_value
        self.current_value = start_value
        self.start_value = start_value
        self.drawdown_history = []
        self.drawdown_start_date = None
        self.days_in_drawdown = 0
        self.alerts = []
        self.is_stopped_out = False


# ================================================================
# PRINT FUNCTIONS
# ================================================================

def print_daily_risk_report(report: DailyRiskReport):
    """Print formatted daily risk report."""
    
    print(f"\n{'─'*60}")
    print(f"📊 DAILY RISK REPORT - {report.date.strftime('%Y-%m-%d')}")
    print(f"{'─'*60}")
    
    print(f"\n💰 P&L:")
    print(f"   Daily:      {report.daily_return:+.2%} (${report.daily_pnl:+,.0f})")
    print(f"   Cumulative: {report.cumulative_return:+.2%} (${report.cumulative_pnl:+,.0f})")
    
    dd_emoji = "🟢" if report.current_drawdown < 0.05 else "🟡" if report.current_drawdown < 0.10 else "🔴"
    print(f"\n📉 Drawdown:")
    print(f"   Current:  {dd_emoji} {report.current_drawdown:.2%} ({report.days_in_drawdown} days)")
    print(f"   Maximum:  {report.max_drawdown:.2%}")
    
    print(f"\n📊 Risk:")
    print(f"   Volatility: {report.realized_volatility:.1%}")
    print(f"   VaR (95%):  {report.var_95:.2%}")
    
    print(f"\n💼 Exposure:")
    print(f"   Net:   {report.net_exposure:.1%}")
    print(f"   Gross: {report.gross_exposure:.1%}")
    
    if report.alerts:
        print(f"\n⚠️  ALERTS ({len(report.alerts)}):")
        for alert in report.alerts:
            emoji = "🔴" if alert.level == AlertLevel.CRITICAL else "🟡"
            print(f"   {emoji} [{alert.category}] {alert.message}")
    
    print(f"{'─'*60}")


def print_risk_alerts(alerts: List[RiskAlert]):
    """Print list of risk alerts."""
    
    if not alerts:
        print("\n✅ No risk alerts")
        return
    
    print(f"\n⚠️  RISK ALERTS ({len(alerts)})")
    print("-" * 60)
    
    for alert in alerts:
        emoji = "🔴" if alert.level == AlertLevel.CRITICAL else "🟡" if alert.level == AlertLevel.WARNING else "🔵"
        print(f"\n{emoji} {alert.timestamp.strftime('%Y-%m-%d')} [{alert.level.value}]")
        print(f"   Category: {alert.category}")
        print(f"   {alert.message}")


def print_risk_summary(summary: Dict):
    """Print risk summary."""
    
    print("\n" + "=" * 60)
    print("📊 POST-TRADE RISK SUMMARY")
    print("=" * 60)
    
    print(f"\n📈 Performance ({summary['n_days']} days):")
    print(f"   Total Return:  {summary['total_return']:+.2%}")
    print(f"   Sharpe Ratio:  {summary['sharpe_ratio']:.2f}")
    print(f"   Win Rate:      {summary['win_rate']:.1%}")
    
    print(f"\n📉 Risk:")
    print(f"   Volatility:    {summary['realized_volatility']:.1%}")
    print(f"   VaR (95%):     {summary['var_95']:.2%}")
    print(f"   Current DD:    {summary['current_drawdown']:.2%}")
    print(f"   Max DD:        {summary['max_drawdown']:.2%}")
    print(f"   Days in DD:    {summary['days_in_drawdown']}")
    
    print(f"\n⚠️  Alerts: {summary['n_alerts']}")
    
    if summary['is_stopped_out']:
        print(f"\n🛑 STATUS: STOPPED OUT")
    else:
        print(f"\n✅ STATUS: Active")
    
    print("=" * 60)
