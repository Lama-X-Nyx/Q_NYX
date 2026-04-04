"""
Central Metrics Library

Single source of truth for all performance metrics.
Used by all validation modules.

Usage:
    from src.validation.metrics import calculate_metrics
    
    metrics = calculate_metrics(trades_df)
    print(metrics['sharpe_ratio'])
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def calculate_metrics(trades: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    """
    Calculate all performance metrics from trades
    
    Handles edge cases:
    - Empty trades
    - Blown up strategies (capital <= 0)
    - Invalid CAGR calculations
    
    Args:
        trades: DataFrame with columns: pnl, pnl_pct, entry_time, exit_time
        initial_capital: Starting capital
    
    Returns:
        Dictionary of all metrics
    """
    
    if len(trades) == 0:
        return _empty_metrics(initial_capital)
    
    # Filter closed trades
    closed = trades[trades['exit_time'].notna()].copy()
    
    if len(closed) == 0:
        return _empty_metrics(initial_capital)
    
    # Basic stats
    total_pnl = closed['pnl'].sum()
    final_capital = initial_capital + total_pnl
    
    # Check for blown up strategy
    is_blown_up = final_capital <= 0
    
    if is_blown_up:
        # Strategy destroyed capital - return special metrics
        return _blown_up_metrics(closed, initial_capital, total_pnl, final_capital)
    
    # Normal metrics calculation
    total_return = (total_pnl / initial_capital) * 100
    
    # Trade stats
    winners = closed[closed['pnl'] > 0]
    losers = closed[closed['pnl'] <= 0]
    
    num_trades = len(closed)
    num_winners = len(winners)
    num_losers = len(losers)
    
    win_rate = (num_winners / num_trades * 100) if num_trades > 0 else 0
    
    # Average trade
    avg_trade = closed['pnl'].mean()
    avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
    
    # Profit factor
    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    # Expectancy
    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
    
    # Equity curve
    closed = closed.sort_values('exit_time')
    closed['cumulative_pnl'] = closed['pnl'].cumsum()
    closed['equity'] = initial_capital + closed['cumulative_pnl']
    
    # Drawdown - cap at -100% minimum
    running_max = closed['equity'].expanding().max()
    drawdown = (closed['equity'] - running_max) / running_max * 100
    max_drawdown = max(drawdown.min(), -100.0)  # Cap at -100%
    
    # Time-based metrics
    closed['exit_time'] = pd.to_datetime(closed['exit_time'])
    closed['entry_time'] = pd.to_datetime(closed['entry_time'])
    
    first_trade = closed['entry_time'].min()
    last_trade = closed['exit_time'].max()
    total_days = (last_trade - first_trade).days
    
    # CAGR - handle edge cases carefully to avoid overflow
    years = total_days / 365.25 if total_days > 0 else 1
    
    # Check ratio before attempting power calculation
    capital_ratio = final_capital / initial_capital if initial_capital > 0 else 0
    
    if final_capital > 0 and years > 0 and capital_ratio > 0:
        # Sanity check BEFORE calculation to avoid overflow
        if capital_ratio > 1000 or capital_ratio < 0.001:
            # Extreme ratio - CAGR would be absurd
            cagr = None
        else:
            try:
                cagr = ((capital_ratio) ** (1 / years) - 1) * 100
                # Final sanity check
                if abs(cagr) > 1000:
                    cagr = None
            except (ValueError, ZeroDivisionError, OverflowError):
                cagr = None
    else:
        cagr = None
    
    # Sharpe ratio (simplified - assuming daily returns)
    if len(closed) > 1:
        daily_returns = closed.groupby(closed['exit_time'].dt.date)['pnl'].sum()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Sortino ratio (downside deviation)
    if len(closed) > 1:
        daily_returns = closed.groupby(closed['exit_time'].dt.date)['pnl'].sum()
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else daily_returns.std()
        sortino_ratio = (daily_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    else:
        sortino_ratio = 0
    
    # Calmar ratio - handle None CAGR
    if cagr is not None and max_drawdown != 0:
        calmar_ratio = (cagr / abs(max_drawdown))
    else:
        calmar_ratio = None
    
    # Exposure time (approximation)
    if total_days > 0:
        trade_durations = (closed['exit_time'] - closed['entry_time']).dt.total_seconds() / 86400
        total_trade_days = trade_durations.sum()
        exposure_time = (total_trade_days / total_days * 100) if total_days > 0 else 0
    else:
        exposure_time = 0
    
    return {
        # Returns
        'total_return': total_return,
        'cagr': cagr,
        'total_pnl': total_pnl,
        
        # Trades
        'total_trades': num_trades,
        'winners': num_winners,
        'losers': num_losers,
        'win_rate': win_rate,
        
        # Trade quality
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        
        # Risk
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        
        # Time
        'exposure_time': exposure_time,
        'total_days': total_days,
        'first_trade': first_trade.isoformat() if pd.notna(first_trade) else None,
        'last_trade': last_trade.isoformat() if pd.notna(last_trade) else None,
        
        # Capital
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'is_blown_up': False
    }


def _blown_up_metrics(closed: pd.DataFrame, initial_capital: float, total_pnl: float, final_capital: float) -> Dict:
    """
    Special metrics for strategies that destroyed capital
    
    Convention: When capital <= 0, strategy is "blown up"
    """
    
    # Basic trade stats still meaningful
    num_trades = len(closed)
    winners = closed[closed['pnl'] > 0]
    losers = closed[closed['pnl'] <= 0]
    
    win_rate = (len(winners) / num_trades * 100) if num_trades > 0 else 0
    
    # Time stats
    closed['exit_time'] = pd.to_datetime(closed['exit_time'])
    closed['entry_time'] = pd.to_datetime(closed['entry_time'])
    first_trade = closed['entry_time'].min()
    last_trade = closed['exit_time'].max()
    total_days = (last_trade - first_trade).days
    
    return {
        # Returns - use special values
        'total_return': (total_pnl / initial_capital) * 100,
        'cagr': None,  # Not meaningful when blown up
        'total_pnl': total_pnl,
        
        # Trades
        'total_trades': num_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        
        # Trade quality
        'avg_trade': closed['pnl'].mean(),
        'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
        'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
        'profit_factor': 0,  # Clearly destroyed capital
        'expectancy': closed['pnl'].mean(),
        
        # Risk - strategy failed
        'max_drawdown': -100.0,  # Convention: blown up = -100%
        'sharpe_ratio': None,
        'sortino_ratio': None,
        'calmar_ratio': None,
        
        # Time
        'exposure_time': 0,
        'total_days': total_days,
        'first_trade': first_trade.isoformat() if pd.notna(first_trade) else None,
        'last_trade': last_trade.isoformat() if pd.notna(last_trade) else None,
        
        # Capital
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'is_blown_up': True
    }


def _empty_metrics(initial_capital: float = 10000) -> Dict:
    """Return empty metrics structure"""
    return {
        'total_return': 0,
        'cagr': None,
        'total_pnl': 0,
        'total_trades': 0,
        'winners': 0,
        'losers': 0,
        'win_rate': 0,
        'avg_trade': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_factor': 0,
        'expectancy': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'calmar_ratio': None,
        'exposure_time': 0,
        'total_days': 0,
        'first_trade': None,
        'last_trade': None,
        'initial_capital': initial_capital,
        'final_capital': initial_capital,
        'is_blown_up': False
    }


def compare_metrics(metrics_a: Dict, metrics_b: Dict, labels: tuple = ('A', 'B')) -> pd.DataFrame:
    """
    Compare two sets of metrics side by side
    
    Args:
        metrics_a: First metrics dict
        metrics_b: Second metrics dict
        labels: Tuple of labels for each metrics set
    
    Returns:
        DataFrame with comparison
    """
    
    df = pd.DataFrame({
        labels[0]: metrics_a,
        labels[1]: metrics_b
    })
    
    # Add difference column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df['Difference'] = df[labels[1]] - df[labels[0]]
        df['% Change'] = ((df[labels[1]] - df[labels[0]]) / df[labels[0]] * 100).replace([np.inf, -np.inf], np.nan)
    
    return df


def metrics_to_dict(metrics: Dict, round_decimals: int = 2) -> Dict:
    """Round all numeric metrics for cleaner output"""
    
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = round(value, round_decimals)
        else:
            result[key] = value
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Metrics Library - Example")
    
    # Create sample trades
    sample_trades = pd.DataFrame({
        'entry_time': pd.date_range('2024-01-01', periods=10, freq='D'),
        'exit_time': pd.date_range('2024-01-02', periods=10, freq='D'),
        'pnl': [100, -50, 150, -30, 200, -40, 80, -20, 120, -60],
        'pnl_pct': [1.0, -0.5, 1.5, -0.3, 2.0, -0.4, 0.8, -0.2, 1.2, -0.6]
    })
    
    # Calculate metrics
    metrics = calculate_metrics(sample_trades, initial_capital=10000)
    
    # Print key metrics
    print(f"\nTotal Return: {metrics['total_return']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
