#!/usr/bin/env python3
"""
NYX Backtest Runner - Run backtests on historical data

Usage:
    python scripts/run_backtest.py --pair BTCUSDT
    python scripts/run_backtest.py --all
    python scripts/run_backtest.py --pair ETHUSDT --start 2020-01-01 --end 2024-01-01
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.nyx_engine import NYXEngine
from src.core.smc import SMCDetector


class SimpleBacktestEngine:
    """
    Backtest engine for NYX v0.8
    
    Features:
    - NYXEngine integration (HSMM + SMC + Macro)
    - Position management
    - Performance tracking
    """
    
    def __init__(self, config: dict, initial_capital: float = 10000):
        self.config = config
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Use NYXEngine (unified signal generation)
        self.engine = NYXEngine(config)
        smc_cfg = config.get('strategy', {}).get('smc', {})
        self.smc = SMCDetector(
            ob_range_threshold=smc_cfg.get('ob_range_threshold', 0.015),
            fvg_min_gap=smc_cfg.get('fvg_min_gap', 0.005),
            liquidity_lookback=smc_cfg.get('liquidity_lookback', 20),
        )

        # State
        self.trades = []
        self.position = None
        self.position_sizes = []
        self.entry_price = 0
        self.entry_sdc = 0
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators and features"""
        
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # ATR
        df['hl_range'] = df['high'] - df['low']
        df['atr_14'] = df['hl_range'].rolling(14).mean()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # Volume
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # Regime
        df['regime'] = 'Range'
        for i in range(200, len(df)):
            price = df['close'].iloc[i]
            sma200 = df['sma_200'].iloc[i]
            sma200_prev = df['sma_200'].iloc[i-20] if i >= 220 else sma200
            
            if pd.isna(sma200):
                continue
            
            slope = (sma200 - sma200_prev) / sma200_prev if sma200_prev != 0 else 0
            
            if price > sma200 and slope > 0.01:
                df.iloc[i, df.columns.get_loc('regime')] = 'Bull'
            elif price < sma200 and slope < -0.01:
                df.iloc[i, df.columns.get_loc('regime')] = 'Bear'
        
        return df
    
    def run(self, pair: str, data: pd.DataFrame) -> dict:
        """
        Run backtest using NYXEngine
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            data: Historical OHLCV DataFrame
        
        Returns:
            Results dictionary
        """
        
        print(f"\n{'='*80}")
        print(f"RUNNING BACKTEST - {pair}")
        print(f"{'='*80}")
        
        # Prepare data
        data = self.prepare_data(data)
        print(f"  Data prepared: {len(data):,} bars")
        print(f"  Period: {data.index[0]} to {data.index[-1]}")
        
        # Run through data
        for idx in range(100, len(data)):  # Start after 100 bars for HSMM
            row = data.iloc[idx]
            current_date = row.name.strftime('%Y-%m-%d')
            current_price = row['close']
            
            # Generate signal from NYXEngine
            signal = self.engine.generate_signal(
                pair=pair,
                data=data.iloc[:idx+1],
                current_date=current_date
            )
            
            # Log signal (every 100 bars to avoid spam)
            if idx % 100 == 0:
                print(f"\n  [{current_date}] {signal['action']}")
                print(f"    Price: ${current_price:,.2f}")
                print(f"    Confidence: {signal['confidence']:.1f}/10")
                print(f"    Regime: {signal['regime']}")
                print(f"    Macro: {signal['macro_signal']} ({signal['macro_strength']:.2f})")
                if signal['reasons']:
                    print(f"    Reasons: {signal['reasons'][0]}")
            
            # Manage existing position
            if self.position:
                self._manage_position_v08(row, signal)
            
            # Check entry
            if not self.position and signal['action'] == 'BUY':
                self._enter_position_v08(pair, row, signal)
        
        # Close any open position
        if self.position:
            self._close_position(data.iloc[-1], "End of backtest")
        
        # Calculate results
        results = self._calculate_results(data)
        results['pair'] = pair
        
        return results
    
    def _get_state_probs(self, window_df: pd.DataFrame) -> np.ndarray:
        """Simplified state probability computation"""
        
        # Check trend
        if len(window_df) < 2:
            return np.array([0.33, 0.34, 0.33])
        
        last_row = window_df.iloc[-1]
        
        if pd.isna(last_row.get('sma_20')) or pd.isna(last_row.get('sma_50')):
            return np.array([0.2, 0.6, 0.2])
        
        # Simple heuristic
        if last_row['sma_20'] > last_row['sma_50'] and last_row['close'] > last_row['sma_20']:
            return np.array([0.65, 0.25, 0.10])  # Trend+
        elif last_row['sma_20'] < last_row['sma_50'] and last_row['close'] < last_row['sma_20']:
            return np.array([0.10, 0.25, 0.65])  # Trend-
        else:
            return np.array([0.20, 0.60, 0.20])  # Range
    
    def _enter_position_v08(self, pair: str, row, signal: dict):
        """Enter position using NYXEngine signal"""
        
        self.position = 'LONG'
        self.entry_price = row['close']
        self.entry_sdc = signal['confidence']
        
        # Position size from signal
        size = signal['position_size']
        self.position_sizes = [size]
        
        # Stops from signal
        self.stop_loss = signal['stop_loss']
        self.take_profit = signal['take_profit']
        
        print(f"\n  🟢 ENTER LONG @ ${row['close']:.2f}")
        print(f"     Size: {size*100:.1f}%")
        print(f"     SL: ${self.stop_loss:.2f}")
        print(f"     TP: ${self.take_profit:.2f}")
        print(f"     Macro: {signal['macro_signal']} ({signal['macro_strength']:.2f})")
    
    def _manage_position_v08(self, row, signal: dict):
        """Manage position using NYXEngine signal"""
        
        current_price = row['close']
        
        # Check stop loss
        if current_price <= self.stop_loss:
            self._close_position(row, "Stop Loss")
            return
        
        # Check take profit
        if current_price >= self.take_profit:
            self._close_position(row, "Take Profit")
            return
        
        # Check exit signal
        if signal['action'] == 'SELL':
            self._close_position(row, f"Exit Signal: {signal['reasons'][0] if signal['reasons'] else 'N/A'}")
            return
    
    def _close_position(self, row, reason: str):
        """Close current position"""
        
        exit_price = row['close']
        pnl_pct = (exit_price - self.entry_price) / self.entry_price
        
        total_size = sum(self.position_sizes)
        pnl = self.capital * pnl_pct * total_size
        
        self.capital += pnl
        
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'type': self.position,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'size': total_size,
            'exit_reason': reason
        })
        
        print(f"\n  🔴 CLOSE {self.position} @ ${exit_price:.2f}")
        print(f"     PnL: ${pnl:+,.2f} ({pnl_pct*100:+.2f}%)")
        print(f"     Reason: {reason}")
        
        self.position = None
        self.position_sizes = []
        self.stop_loss = None
        self.take_profit = None
    
    def _check_entry(self, row, confidence, state_probs, recent_data):
        """Check for entry signal"""
        
        sdc_threshold = self.config.get('strategy', {}).get('hsmm', {}).get('sdc_threshold', 3.5)
        prob_threshold = self.config.get('strategy', {}).get('hsmm', {}).get('prob_threshold', 0.55)
        
        if confidence < sdc_threshold:
            return
        
        # Detect SMC patterns
        patterns = self.smc.detect_all(recent_data)
        
        # LONG signal
        if state_probs[0] > prob_threshold and patterns.get('bullish_fvg', False):
            self.position = 'LONG'
            self.entry_price = row['close']
            self.entry_sdc = confidence
            
            # Position size
            size = self._calculate_position_size(confidence, row['regime'])
            self.position_sizes = [size]
        
        # SHORT signal
        elif state_probs[2] > prob_threshold and patterns.get('bearish_fvg', False):
            self.position = 'SHORT'
            self.entry_price = row['close']
            self.entry_sdc = confidence
            size = 0.03  # Conservative for shorts
            self.position_sizes = [size]
    
    def _manage_position(self, row, confidence, state_probs):
        """Manage existing position"""
        
        pnl_pct = (row['close'] - self.entry_price) / self.entry_price
        if self.position == 'SHORT':
            pnl_pct = -pnl_pct
        
        total_size = sum(self.position_sizes)
        pnl = self.capital * pnl_pct * total_size
        
        # Pyramiding
        pyramid_threshold = self.config.get('risk', {}).get('pyramid_threshold', 0.10)
        if (self.position == 'LONG' and pnl_pct > pyramid_threshold and 
            self.entry_sdc > 6 and len(self.position_sizes) < 3):
            pyramid_size = self.position_sizes[0] * 0.5
            self.position_sizes.append(pyramid_size)
        
        # Exit conditions
        stop_loss = self.config.get('risk', {}).get('stop_loss', 0.05)
        
        should_exit = False
        exit_reason = ""
        
        # Stop loss
        if pnl_pct < -stop_loss:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Reverse signal
        elif self.position == 'LONG' and state_probs[2] > 0.55:  # Trend- signal
            should_exit = True
            exit_reason = "reverse_signal"
        elif self.position == 'SHORT' and state_probs[0] > 0.55:  # Trend+ signal
            should_exit = True
            exit_reason = "reverse_signal"
        
        if should_exit:
            self.capital += pnl
            
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': row['close'],
                'type': self.position,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'size': total_size,
                'exit_reason': exit_reason
            })
            
            self.position = None
            self.position_sizes = []
    
    def _calculate_position_size(self, sdc, regime):
        """Dynamic position sizing"""
        
        base_size = self.config.get('risk', {}).get('base_size', 0.05)
        max_position = self.config.get('risk', {}).get('max_position', 0.25)
        
        # Scale by confidence
        if sdc >= 8.5:
            base = 0.10
        elif sdc >= 7.5:
            base = 0.07
        elif sdc >= 6.5:
            base = 0.05
        else:
            base = 0.03
        
        # Regime multiplier
        if regime == 'Bull':
            multiplier = self.config.get('risk', {}).get('bull_multiplier', 2.5)
        elif regime == 'Bear':
            multiplier = self.config.get('risk', {}).get('bear_multiplier', 0.7)
        else:
            multiplier = 1.0
        
        size = min(base * multiplier, max_position)
        
        return size
    
    def _calculate_results(self, data):
        """Calculate performance metrics"""
        
        if not self.trades:
            return {
                'error': 'No trades generated',
                'initial_capital': self.initial_capital,
                'final_capital': self.capital
            }
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        winners = sum(1 for t in self.trades if t['pnl'] > 0)
        win_rate = winners / len(self.trades) * 100
        
        years = (data.index[-1] - data.index[0]).days / 365.25
        cagr = ((self.capital / self.initial_capital)**(1/years) - 1) * 100
        
        avg_win = np.mean([t['pnl_pct'] for t in self.trades if t['pnl'] > 0]) if winners > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in self.trades if t['pnl'] < 0]) if winners < len(self.trades) else 0
        
        profit_factor = (avg_win * winners / (avg_loss * (len(self.trades) - winners))) if avg_loss > 0 and winners < len(self.trades) else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'cagr': cagr,
            'total_trades': len(self.trades),
            'winners': winners,
            'losers': len(self.trades) - winners,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'years': years
        }
        
        return results


def load_config():
    """Load configuration"""
    
    config_path = 'config/config.yaml'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default config
        return {
            'strategy': {
                'hsmm': {'sdc_threshold': 3.5, 'prob_threshold': 0.55},
                'smc': {'ob_range_threshold': 0.015}
            },
            'risk': {
                'base_size': 0.05,
                'max_position': 0.25,
                'stop_loss': 0.05,
                'pyramid_threshold': 0.10,
                'bull_multiplier': 2.5,
                'bear_multiplier': 0.7
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Run NYX backtests')
    
    parser.add_argument('--pair', type=str, help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--all', action='store_true', help='Run all enabled pairs')
    parser.add_argument('--output', type=str, default='data/results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    print("\n" + "█"*80)
    print("NYX v0.8 BACKTEST RUNNER")
    print("█"*80)
    
    if args.all:
        # Load pairs config
        pairs_path = 'config/pairs.yaml'
        if os.path.exists(pairs_path):
            with open(pairs_path, 'r') as f:
                pairs_config = yaml.safe_load(f)
            pairs = [k for k, v in pairs_config.get('pairs', {}).items() if v.get('enabled', False)]
        else:
            pairs = ['BTCUSDT', 'ETHUSDT']
        
        print(f"\nRunning backtests on {len(pairs)} pairs:")
        for pair in pairs:
            print(f"  • {pair}")
        
        all_results = {}
        
        for pair in pairs:
            # Load data
            data_file = f"data/raw/{pair}_1h.csv"
            
            if not os.path.exists(data_file):
                print(f"\n⚠ Data not found for {pair}: {data_file}")
                continue
            
            print(f"\n{'='*80}")
            print(f"BACKTEST: {pair}")
            print(f"{'='*80}")
            
            data = pd.read_csv(data_file)
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.set_index('datetime')
            
            # Filter by date if specified
            if args.start:
                data = data[data.index >= args.start]
            if args.end:
                data = data[data.index <= args.end]
            
            data_df: pd.DataFrame = pd.DataFrame(data)
            print(f"  Period: {data_df.index[0]} → {data_df.index[-1]}")
            print(f"  Bars: {len(data_df):,}")

            # Run backtest
            engine = SimpleBacktestEngine(config, args.capital)
            results = engine.run(pair, data_df)
            
            all_results[pair] = results
            
            # Print results
            print(f"\n  RESULTS:")
            if 'error' in results:
                print(f"    ✗ {results['error']}")
            else:
                print(f"    Return:        {results['total_return']:+.2f}%")
                print(f"    CAGR:          {results['cagr']:.2f}%/year")
                print(f"    Trades:        {results['total_trades']}")
                print(f"    Win Rate:      {results['win_rate']:.1f}%")
                print(f"    Profit Factor: {results['profit_factor']:.2f}")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY - ALL PAIRS")
        print(f"{'='*80}")
        print(f"\n{'Pair':<12} {'Return':>10} {'CAGR':>10} {'Trades':>8} {'Win Rate':>10}")
        print("-"*80)
        
        for pair, res in all_results.items():
            if 'error' not in res:
                print(f"{pair:<12} {res['total_return']:>9.2f}% {res['cagr']:>9.2f}% {res['total_trades']:>8} {res['win_rate']:>9.1f}%")
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"backtest_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
    
    elif args.pair:
        # Single pair backtest
        data_file = f"data/raw/{args.pair}_1h.csv"
        
        if not os.path.exists(data_file):
            print(f"\n✗ Data not found: {data_file}")
            print(f"  Run: python scripts/download_data.py --pair {args.pair}")
            return
        
        print(f"\nLoading data: {data_file}")
        data = pd.read_csv(data_file)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        
        if args.start:
            data = data[data.index >= args.start]
        if args.end:
            data = data[data.index <= args.end]
        
        data_df2: pd.DataFrame = pd.DataFrame(data)
        print(f"  Period: {data_df2.index[0]} → {data_df2.index[-1]}")
        print(f"  Bars: {len(data_df2):,}")

        # Run backtest
        engine = SimpleBacktestEngine(config, args.capital)
        results = engine.run(args.pair, data_df2)
        
        # Print results
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        
        if 'error' in results:
            print(f"\n✗ {results['error']}")
        else:
            print(f"\nPerformance:")
            print(f"  Initial Capital:  ${results['initial_capital']:,.2f}")
            print(f"  Final Capital:    ${results['final_capital']:,.2f}")
            print(f"  Total Return:     {results['total_return']:+.2f}%")
            print(f"  CAGR:             {results['cagr']:.2f}%/year")
            print(f"\nTrades:")
            print(f"  Total:            {results['total_trades']}")
            print(f"  Winners:          {results['winners']}")
            print(f"  Losers:           {results['losers']}")
            print(f"  Win Rate:         {results['win_rate']:.1f}%")
            print(f"\nMetrics:")
            print(f"  Avg Win:          {results['avg_win']:+.2f}%")
            print(f"  Avg Loss:         -{results['avg_loss']:.2f}%")
            print(f"  Profit Factor:    {results['profit_factor']:.2f}")
            print(f"  Period:           {results['years']:.1f} years")
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"backtest_{args.pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
