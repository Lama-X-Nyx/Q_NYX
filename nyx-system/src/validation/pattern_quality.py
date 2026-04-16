"""
Pattern Quality Attribution

Measures trade quality by pattern source: FVG-only vs OB-assisted.

Goal: Determine if OB adds value despite low coverage.

Usage:
    from src.validation.pattern_quality import PatternQuality
    
    pq = PatternQuality(config)
    report = pq.run(data, pair='BTCUSDT')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, cast
import json
from pathlib import Path


class PatternQuality:
    """
    Pattern Quality Attribution
    
    Classifies trades by pattern source and measures quality
    """
    
    def __init__(self, config: Dict):
        """
        Initialize pattern quality analyzer
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Trade groups
        self.groups = {
            'fvg_only': [],
            'ob_only': [],
            'fvg_and_ob': [],
            'no_pattern': []
        }
    
    def run(self, data: pd.DataFrame, pair: str, quiet: bool = False) -> Dict:
        """
        Run pattern quality analysis
        
        Args:
            data: OHLCV DataFrame
            pair: Trading pair
            quiet: Suppress output
        
        Returns:
            Quality report dict
        """
        
        if not quiet:
            print(f"\n{'='*80}")
            print(f"PATTERN QUALITY - {pair}")
            print(f"{'='*80}")
        
        from src.core.nyx_engine_v08 import NYXEngine  # legacy v0.8 (Ticket 04)

        # Initialize engine
        engine = NYXEngine(self.config)
        
        # Prepare data
        prepared_data = engine.prepare_data(data)
        
        # Get config
        val_config = self.config.get('validation', {})
        warmup = val_config.get('warmup_bars', 100)
        lookback = val_config.get('signal_lookback_bars', 50)
        
        if not quiet:
            print(f"\nAnalyzing {len(prepared_data)} bars...")
            print(f"Warmup: {warmup} bars")
            print(f"Lookback: {lookback} bars")
        
        # Simulate trading and classify
        trades = []
        position = None
        capital = 10000
        
        for i in range(warmup, len(prepared_data)):
            current_date = cast(pd.Timestamp, prepared_data.index[i])
            
            # Get signal
            signal = engine.generate_signal_at_index(
                pair=pair,
                prepared_data=prepared_data,
                i=i,
                current_date=current_date.isoformat(),
                lookback_bars=lookback
            )
            
            # Track trades
            if signal['action'] == 'BUY' and position is None:
                position = {
                    'entry_time': current_date,
                    'entry_price': prepared_data.iloc[i]['close'],
                    'entry_signal': signal,
                    'side': 'long'
                }
            
            elif signal['action'] == 'SELL' and position is not None:
                # Close long position
                exit_price = prepared_data.iloc[i]['close']
                pnl = (exit_price - position['entry_price']) / position['entry_price'] * 100
                
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_date,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'duration_bars': (current_date - position['entry_time']).total_seconds() / 3600,
                    'entry_signal': position['entry_signal']
                }
                
                trades.append(trade)
                position = None
        
        # Close open position
        if position is not None:
            exit_price = prepared_data.iloc[-1]['close']
            pnl = (exit_price - position['entry_price']) / position['entry_price'] * 100
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': prepared_data.index[-1],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl_pct': pnl,
                'duration_bars': (prepared_data.index[-1] - position['entry_time']).total_seconds() / 3600,
                'entry_signal': position['entry_signal']
            }
            
            trades.append(trade)
        
        # Classify trades
        self._classify_trades(trades)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate report
        report = {
            'pair': pair,
            'bars_analyzed': len(prepared_data),
            'total_trades': len(trades),
            'trade_groups': metrics,
            'summary': self._generate_summary(metrics)
        }
        
        return report
    
    def _classify_trades(self, trades: List[Dict]):
        """
        Classify trades by pattern source
        
        Args:
            trades: List of trade dicts
        """
        
        for trade in trades:
            signal = trade['entry_signal']
            smc_patterns = signal.get('smc_patterns', {})
            
            # Check patterns
            has_bullish_fvg = smc_patterns.get('bullish_fvg', False)
            has_bearish_fvg = smc_patterns.get('bearish_fvg', False)
            has_bullish_ob = smc_patterns.get('bullish_ob', False)
            has_bearish_ob = smc_patterns.get('bearish_ob', False)
            
            has_fvg = has_bullish_fvg or has_bearish_fvg
            has_ob = has_bullish_ob or has_bearish_ob
            
            # Classify
            if has_fvg and has_ob:
                self.groups['fvg_and_ob'].append(trade)
            elif has_fvg:
                self.groups['fvg_only'].append(trade)
            elif has_ob:
                self.groups['ob_only'].append(trade)
            else:
                self.groups['no_pattern'].append(trade)
    
    def _calculate_metrics(self) -> Dict:
        """Calculate metrics for each group"""
        
        metrics = {}
        
        for group_name, trades in self.groups.items():
            if len(trades) == 0:
                metrics[group_name] = {
                    'count': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'profit_factor': 0,
                    'expectancy': 0,
                    'max_drawdown': 0,
                    'avg_duration': 0
                }
                continue
            
            # Extract PnLs
            pnls = [t['pnl_pct'] for t in trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            
            # Calculate metrics
            win_rate = (len(winners) / len(pnls)) * 100 if pnls else 0
            avg_pnl = np.mean(pnls) if pnls else 0
            
            # Profit factor
            total_wins = sum(winners) if winners else 0
            total_losses = abs(sum(losers)) if losers else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else (float('inf') if total_wins > 0 else 0)
            
            # Expectancy
            expectancy = avg_pnl
            
            # Max drawdown (cumulative)
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            # Avg duration
            durations = [t['duration_bars'] for t in trades]
            avg_duration = np.mean(durations) if durations else 0
            
            metrics[group_name] = {
                'count': len(trades),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'profit_factor': profit_factor if profit_factor != float('inf') else 999,
                'expectancy': expectancy,
                'max_drawdown': max_drawdown,
                'avg_duration': avg_duration
            }
        
        return metrics
    
    def _generate_summary(self, metrics: Dict) -> str:
        """Generate text summary"""
        
        summary = f"\nTrades analyzed: {sum(m['count'] for m in metrics.values())}\n\n"
        
        for group_name, m in metrics.items():
            if m['count'] == 0:
                continue
            
            summary += f"{group_name.upper().replace('_', ' ')}:\n"
            summary += f"  Trades:         {m['count']:6d}\n"
            summary += f"  Win rate:       {m['win_rate']:6.1f}%\n"
            summary += f"  Avg PnL:        {m['avg_pnl']:6.2f}%\n"
            summary += f"  Profit factor:  {m['profit_factor']:6.2f}\n"
            summary += f"  Expectancy:     {m['expectancy']:6.2f}%\n"
            summary += f"  Max DD:         {m['max_drawdown']:6.2f}%\n"
            summary += f"  Avg duration:   {m['avg_duration']:6.1f}h\n\n"
        
        # Determine best quality
        non_empty = {k: v for k, v in metrics.items() if v['count'] > 0}
        
        if non_empty:
            best_expectancy = max(non_empty.items(), key=lambda x: x[1]['expectancy'])
            highest_count = max(non_empty.items(), key=lambda x: x[1]['count'])
            
            summary += "Conclusion:\n"
            summary += f"  Best quality: {best_expectancy[0]} (expectancy: {best_expectancy[1]['expectancy']:.2f}%)\n"
            summary += f"  Most trades: {highest_count[0]} ({highest_count[1]['count']} trades)\n"
        
        return summary


def save_pattern_quality(report: Dict, output_dir: str):
    """
    Save pattern quality report to JSON
    
    Args:
        report: Quality report dict
        output_dir: Output directory
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pair = report.get('pair', 'UNKNOWN')
    json_file = output_path / f"{pair}_pattern_quality.json"
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Pattern quality report saved: {json_file}")


if __name__ == "__main__":
    print("Pattern Quality Attribution")
    
    import yaml
    
    # Load config
    with open('config/validation_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set min_required_bars to 50
    config['strategy']['min_required_bars'] = 50
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    # Run analysis
    pq = PatternQuality(config)
    report = pq.run(data, pair='BTCUSDT')
    
    print(report['summary'])
