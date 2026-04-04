#!/usr/bin/env python3
"""
NYX Paper Trading Runner v0.8

Live paper trading with:
- NYXEngine (HSMM + SMC + Macro)
- Binance live data
- SQLite logging
- Position management

Usage:
    python src/runner/run_paper.py
    python src/runner/run_paper.py --pairs BTCUSDT,ETHUSDT --capital 50000
    python src/runner/run_paper.py --duration 24 --interval 1h
"""

import sys
import argparse
import time
import yaml
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.nyx_engine import NYXEngine
from src.execution.paper_engine import PaperEngine
from src.data.live_feed import BinanceFeed


class PaperTradingRunner:
    """
    Live paper trading runner
    
    Combines:
    - NYXEngine (signals)
    - PaperEngine (execution)
    - BinanceFeed (data)
    """
    
    def __init__(
        self, 
        pairs: list,
        initial_capital: float = 10000,
        interval: str = '1h',
        update_seconds: int = 300
    ):
        """
        Initialize runner
        
        Args:
            pairs: Trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            initial_capital: Starting capital
            interval: Candle interval ('1h', '4h', etc.)
            update_seconds: Seconds between updates
        """
        self.pairs = pairs
        self.interval = interval
        self.update_seconds = update_seconds
        
        # Load config
        config_path = 'config/config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        self.engine = NYXEngine(config)
        self.paper = PaperEngine(initial_capital=initial_capital)
        self.feed = BinanceFeed(pairs=pairs)
        
        # Stats
        self.iteration = 0
        self.start_time = datetime.now()
        
        print("\n" + "█"*80)
        print("NYX PAPER TRADING v0.8 - LIVE")
        print("█"*80)
        print(f"Pairs:    {', '.join(pairs)}")
        print(f"Capital:  ${initial_capital:,.2f}")
        print(f"Interval: {interval}")
        print(f"Update:   {update_seconds}s")
        print("█"*80 + "\n")
    
    def run(self, duration_hours: float = None):
        """
        Run paper trading
        
        Args:
            duration_hours: How long to run (None = forever)
        """
        
        try:
            while True:
                self.iteration += 1
                
                print(f"\n{'='*80}")
                print(f"ITERATION #{self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")
                
                # Process each pair
                for pair in self.pairs:
                    self._process_pair(pair)
                
                # Print status
                self.paper.print_status()
                
                # Check duration
                if duration_hours:
                    elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        print(f"\n✓ Duration complete ({duration_hours}h)")
                        break
                
                # Wait
                print(f"\n⏳ Waiting {self.update_seconds}s until next update...")
                time.sleep(self.update_seconds)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        finally:
            self._shutdown()
    
    def _process_pair(self, pair: str):
        """Process single trading pair"""
        
        print(f"\n📈 Processing {pair}...")
        
        # Get current price
        current_price = self.feed.get_price(pair)
        if not current_price:
            print(f"   ⚠️ Could not fetch price")
            return
        
        print(f"   Price: ${current_price:,.2f}")
        
        # Get recent data for HSMM
        data = self.feed.get_recent_candles(pair, interval=self.interval, limit=200)
        if data is None or len(data) < 100:
            print(f"   ⚠️ Insufficient data ({len(data) if data is not None else 0} candles)")
            return
        
        # Generate signal
        current_date = datetime.now().strftime('%Y-%m-%d')
        signal = self.engine.generate_signal(pair, data, current_date)
        
        print(f"   Signal: {signal['action']}")
        print(f"   Confidence: {signal['confidence']:.1f}/10")
        print(f"   Regime: {signal['regime']}")
        print(f"   Macro: {signal['macro_signal']} ({signal['macro_strength']:.2f})")
        
        if signal['reasons']:
            print(f"   Reason: {signal['reasons'][0]}")
        
        # Execute signal
        action = self.paper.execute_signal(pair, signal, current_price)
        
        if action and action != 'HOLD':
            print(f"   → Action: {action}")
    
    def _shutdown(self):
        """Graceful shutdown"""
        
        print("\n" + "="*80)
        print("SHUTTING DOWN")
        print("="*80)
        
        # Final status
        self.paper.print_status()
        
        # Get stats
        stats = self.paper.get_stats()
        
        print("\n📊 FINAL STATISTICS:")
        print(f"   Total Trades:  {stats['total_trades']}")
        print(f"   Win Rate:      {stats['win_rate']:.1f}%")
        print(f"   Total Return:  {stats['total_return']:+.2f}%")
        print(f"   Avg Win:       ${stats['avg_win']:+,.2f}")
        print(f"   Avg Loss:      ${stats['avg_loss']:+,.2f}")
        
        # Close
        self.paper.close()
        
        print("\n✓ Shutdown complete")
        print(f"Database: data/paper_trading.db")


def main():
    parser = argparse.ArgumentParser(description='NYX Paper Trading v0.8')
    
    parser.add_argument(
        '--pairs',
        type=str,
        default='BTCUSDT,ETHUSDT',
        help='Comma-separated pairs (e.g., BTCUSDT,ETHUSDT)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital (USD)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Candle interval'
    )
    
    parser.add_argument(
        '--update',
        type=int,
        default=300,
        help='Update interval (seconds)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Duration in hours (None = continuous)'
    )
    
    args = parser.parse_args()
    
    # Parse pairs
    pairs = [p.strip() for p in args.pairs.split(',')]
    
    # Create runner
    runner = PaperTradingRunner(
        pairs=pairs,
        initial_capital=args.capital,
        interval=args.interval,
        update_seconds=args.update
    )
    
    # Run
    runner.run(duration_hours=args.duration)


if __name__ == "__main__":
    main()
