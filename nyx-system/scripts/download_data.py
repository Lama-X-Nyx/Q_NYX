#!/usr/bin/env python3
"""
NYX Data Downloader - Download historical crypto data from Binance

Usage:
    python scripts/download_data.py --pair BTCUSDT --timeframes 15m,1h,4h --since 2020-01-01
    python scripts/download_data.py --all
"""

import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class BinanceDownloader:
    """Download OHLCV data from Binance public API"""
    
    def __init__(self, base_url: str = 'https://api.binance.com'):
        self.base_url = base_url
        self.session = requests.Session()
    
    def download(self, symbol: str, interval: str, since: str, output_dir: str = 'data/raw'):
        """
        Download historical OHLCV data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            since: Start date 'YYYY-MM-DD'
            output_dir: Directory to save CSV files
        """
        print(f"\n{'='*80}")
        print(f"Downloading {symbol} {interval}")
        print(f"{'='*80}")
        
        # Convert since to timestamp
        start_dt = datetime.strptime(since, '%Y-%m-%d')
        start_ts = int(start_dt.timestamp() * 1000)
        
        all_klines = []
        current_ts = start_ts
        
        endpoint = f"{self.base_url}/api/v3/klines"
        
        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ts,
                'limit': 1000
            }
            
            try:
                response = self.session.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                
                klines = response.json()
                
                if not klines:
                    print(f"  ✓ No more data available")
                    break
                
                all_klines.extend(klines)
                
                # Update timestamp
                current_ts = klines[-1][0] + 1
                
                # Check if reached present
                last_dt = datetime.fromtimestamp(klines[-1][0] / 1000)
                
                print(f"  Downloaded {len(all_klines):,} candles (last: {last_dt.strftime('%Y-%m-%d %H:%M')})")
                
                if last_dt >= datetime.now() - timedelta(hours=1):
                    print(f"  ✓ Reached present time")
                    break
                
                # Rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"  ✗ Error: {e}")
                break
        
        if not all_klines:
            print(f"  ✗ No data downloaded")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Select and convert columns
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Reorder columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{symbol}_{interval}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        print(f"\n  ✓ Saved {len(df):,} candles to {filepath}")
        print(f"  Period: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return filepath
    
    def download_multi_timeframe(self, symbol: str, timeframes: list, since: str, output_dir: str = 'data/raw'):
        """Download multiple timeframes for a pair"""
        
        print(f"\n{'█'*80}")
        print(f"MULTI-TIMEFRAME DOWNLOAD: {symbol}")
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Since: {since}")
        print(f"{'█'*80}")
        
        results = {}
        
        for tf in timeframes:
            filepath = self.download(symbol, tf, since, output_dir)
            if filepath:
                results[tf] = filepath
            time.sleep(0.5)  # Rate limit between timeframes
        
        return results


def load_pairs_config():
    """Load pairs from config/pairs.yaml"""
    import yaml
    
    config_path = 'config/pairs.yaml'
    
    if not os.path.exists(config_path):
        print(f"⚠ Config file not found: {config_path}")
        print(f"  Using default pairs: BTCUSDT, ETHUSDT")
        return {
            'BTCUSDT': {'enabled': True, 'timeframes': ['15m', '1h', '4h']},
            'ETHUSDT': {'enabled': True, 'timeframes': ['15m', '1h', '4h']}
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('pairs', {})


def main():
    parser = argparse.ArgumentParser(description='Download crypto data from Binance')
    
    parser.add_argument('--pair', type=str, help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('--timeframes', type=str, help='Comma-separated timeframes (e.g., 15m,1h,4h)')
    parser.add_argument('--since', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Download all enabled pairs from config')
    
    args = parser.parse_args()
    
    downloader = BinanceDownloader()
    
    if args.all:
        # Download all enabled pairs
        print("\n" + "█"*80)
        print("DOWNLOADING ALL ENABLED PAIRS")
        print("█"*80)
        
        pairs_config = load_pairs_config()
        enabled_pairs = {k: v for k, v in pairs_config.items() if v.get('enabled', False)}
        
        if not enabled_pairs:
            print("\n✗ No enabled pairs found in config/pairs.yaml")
            return
        
        print(f"\nFound {len(enabled_pairs)} enabled pairs:")
        for pair in enabled_pairs.keys():
            print(f"  • {pair}")
        
        for pair, config in enabled_pairs.items():
            timeframes = config.get('timeframes', ['15m', '1h', '4h'])
            downloader.download_multi_timeframe(
                symbol=pair,
                timeframes=timeframes,
                since=args.since,
                output_dir=args.output_dir
            )
            time.sleep(1)  # Rate limit between pairs
        
        print("\n" + "="*80)
        print("✅ ALL DOWNLOADS COMPLETE")
        print("="*80)
    
    elif args.pair:
        # Download single pair
        if args.timeframes:
            timeframes = args.timeframes.split(',')
            downloader.download_multi_timeframe(
                symbol=args.pair,
                timeframes=timeframes,
                since=args.since,
                output_dir=args.output_dir
            )
        else:
            # Single timeframe (default 1h)
            downloader.download(
                symbol=args.pair,
                interval='1h',
                since=args.since,
                output_dir=args.output_dir
            )
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/download_data.py --pair BTCUSDT --timeframes 15m,1h,4h --since 2020-01-01")
        print("  python scripts/download_data.py --pair ETHUSDT --since 2021-01-01")
        print("  python scripts/download_data.py --all")


if __name__ == "__main__":
    main()
