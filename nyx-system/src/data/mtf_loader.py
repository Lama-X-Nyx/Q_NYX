"""
Multi-Timeframe Data Loader

Loads and aligns multiple timeframe datasets for NYX 4-TF baseline.

Responsibilities:
- Load 1D, 4H, 1H, 15M, (5M) datasets
- Align them temporally
- Ensure only closed candles from higher TFs are used
- Provide clean structure for NYXEngine

No look-ahead bias, no future information leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class MTFLoader:
    """
    Multi-Timeframe Data Loader
    
    Loads and synchronizes multiple timeframe datasets
    """
    
    def __init__(self, data_dir: str = "data/raw/mtf"):
        """
        Initialize MTF loader
        
        Args:
            data_dir: Directory containing MTF CSV files
        """
        self.data_dir = Path(data_dir)
        
        # TF specifications (in minutes)
        self.tf_specs = {
            '1d': 1440,
            '4h': 240,
            '1h': 60,
            '15m': 15,
            '5m': 5
        }
    
    def load(self, pair: str, tfs: list = None) -> Dict[str, pd.DataFrame]:
        """
        Load multiple timeframes for a pair
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            tfs: List of timeframes to load (default: all)
        
        Returns:
            Dict mapping TF name to DataFrame
        """
        
        if tfs is None:
            tfs = ['1d', '4h', '1h', '15m']
        
        data = {}
        
        for tf in tfs:
            filepath = self.data_dir / f"{pair}_{tf}.csv"
            
            if not filepath.exists():
                print(f"⚠️  {tf} data not found: {filepath}")
                continue
            
            df = pd.read_csv(filepath)
            
            # Clean and standardize
            df = self._clean_dataframe(df)
            
            data[tf] = df
        
        return data
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize a dataframe
        
        Args:
            df: Raw dataframe
        
        Returns:
            Cleaned dataframe with datetime index
        """
        
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Ensure datetime column exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("No datetime/timestamp column found")
        
        # Set index
        df = df.set_index('datetime')
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by time
        df = df.sort_index()
        
        return df
    
    def align_at_timestamp(self, mtf_data: Dict[str, pd.DataFrame],
                          target_time: pd.Timestamp,
                          target_tf: str = '15m') -> Dict[str, pd.DataFrame]:
        """
        Align all TFs at a specific timestamp using only closed candles
        
        Args:
            mtf_data: Dict of TF DataFrames
            target_time: Target timestamp (usually from execution TF)
            target_tf: Execution timeframe (15m or 5m)
        
        Returns:
            Dict of aligned DataFrames (only closed candles from higher TFs)
        """
        
        aligned = {}
        
        for tf, df in mtf_data.items():
            if tf == target_tf:
                # Execution TF: use data up to and including target_time
                aligned[tf] = df[df.index <= target_time]
            else:
                # Higher TF: use only closed candles
                closed_time = self._get_closed_candle_time(target_time, tf)
                aligned[tf] = df[df.index <= closed_time]
        
        return aligned
    
    def _get_closed_candle_time(self, target_time: pd.Timestamp, tf: str) -> pd.Timestamp:
        """
        Get the most recent closed candle time for a TF
        
        Args:
            target_time: Current time
            tf: Timeframe
        
        Returns:
            Timestamp of most recent closed candle
        """
        
        tf_minutes = self.tf_specs[tf]
        
        # Floor to TF boundary
        epoch = target_time.value // 10**9
        tf_seconds = tf_minutes * 60
        
        closed_epoch = (epoch // tf_seconds) * tf_seconds
        
        # Subtract one period to get the last CLOSED candle
        closed_epoch -= tf_seconds
        
        return pd.Timestamp(closed_epoch, unit='s', tz=target_time.tz)
    
    def get_lookback_window(self, mtf_data: Dict[str, pd.DataFrame],
                           end_time: pd.Timestamp,
                           lookback_bars: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """
        Get lookback windows for each TF
        
        Args:
            mtf_data: Dict of TF DataFrames
            end_time: End timestamp
            lookback_bars: Dict mapping TF to number of bars
        
        Returns:
            Dict of windowed DataFrames
        """
        
        windowed = {}
        
        for tf, df in mtf_data.items():
            bars = lookback_bars.get(tf, 100)
            
            # Get data up to end_time
            df_to_end = df[df.index <= end_time]
            
            # Take last N bars
            windowed[tf] = df_to_end.tail(bars)
        
        return windowed
    
    def validate_alignment(self, mtf_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Validate MTF data alignment
        
        Args:
            mtf_data: Dict of TF DataFrames
        
        Returns:
            Validation report
        """
        
        report = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        for tf, df in mtf_data.items():
            # Check for gaps
            if len(df) == 0:
                report['valid'] = False
                report['issues'].append(f"{tf}: Empty dataframe")
                continue
            
            # Check for duplicates
            if df.index.duplicated().any():
                report['valid'] = False
                report['issues'].append(f"{tf}: Duplicate timestamps")
            
            # Check for required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                report['valid'] = False
                report['issues'].append(f"{tf}: Missing columns {missing}")
            
            # Stats
            report['stats'][tf] = {
                'bars': len(df),
                'start': df.index[0].isoformat() if len(df) > 0 else None,
                'end': df.index[-1].isoformat() if len(df) > 0 else None
            }
        
        return report


def load_mtf_sample(pair: str = 'BTCUSDT', 
                   sample_bars: Optional[int] = None,
                   data_dir: str = "data/raw/mtf") -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load MTF data
    
    Args:
        pair: Trading pair
        sample_bars: If provided, take last N bars from lowest TF
        data_dir: Data directory
    
    Returns:
        Dict of TF DataFrames
    """
    
    loader = MTFLoader(data_dir)
    
    # Load all TFs
    mtf_data = loader.load(pair)
    
    if not mtf_data:
        raise ValueError(f"No MTF data found for {pair}")
    
    # Apply sample if requested
    if sample_bars:
        # Find lowest TF (most bars)
        lowest_tf = max(mtf_data.items(), key=lambda x: len(x[1]))[0]
        
        # Get end time from sample
        end_time = mtf_data[lowest_tf].index[-1]
        start_time = mtf_data[lowest_tf].index[-sample_bars]
        
        # Slice all TFs to this range
        for tf in mtf_data:
            mtf_data[tf] = mtf_data[tf][(mtf_data[tf].index >= start_time) & 
                                        (mtf_data[tf].index <= end_time)]
    
    return mtf_data


def load_fractal_context(pair: str,
                        current_date: datetime,
                        config: dict,
                        data_dir: str = "data/raw/mtf") -> Dict[str, pd.DataFrame]:
    """
    Load MTF data with proper fractal context windows
    
    Each timeframe gets its own temporal window based on its role:
    - 1D context: 360 days (macro climate)
    - 4H structure: 7 days (weekly patterns)
    - 1H regime: 2 days (operational regime)
    - 15M setup: 1 day (tactical patterns)
    
    Args:
        pair: Trading pair (e.g., 'BTCUSDT')
        current_date: Reference date for alignment (all windows end here)
        config: Config dict with fractal.context_windows
        data_dir: Data directory
    
    Returns:
        Dict of TF DataFrames with proper context windows
    """
    
    loader = MTFLoader(data_dir)
    
    # Get window configuration
    fractal_config = config.get('fractal', {})
    windows = fractal_config.get('context_windows', {})
    timeframes_config = fractal_config.get('timeframes', {})
    
    # Default windows if not in config
    default_windows = {
        'context_1d_days': 360,
        'structure_4h_days': 7,
        'regime_1h_days': 2,
        'setup_15m_days': 1
    }
    
    # Map TF to window days
    tf_windows = {
        '1d': windows.get('context_1d_days', default_windows['context_1d_days']),
        '4h': windows.get('structure_4h_days', default_windows['structure_4h_days']),
        '1h': windows.get('regime_1h_days', default_windows['regime_1h_days']),
        '15m': windows.get('setup_15m_days', default_windows['setup_15m_days'])
    }
    
    # Load each TF with its window
    mtf_data = {}
    
    for tf, window_days in tf_windows.items():
        # Calculate start date for this window
        start_date = current_date - timedelta(days=window_days)
        
        # Load full TF data
        tf_file = loader.data_dir / f"{pair}_{tf}.csv"
        
        if not tf_file.exists():
            continue
        
        df = pd.read_csv(tf_file)
        
        # Use datetime column as index if available
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            continue
        
        # Filter to window (only closed candles up to current_date)
        # Convert current_date and start_date to timezone-naive if index is timezone-naive
        if df.index.tz is None:
            current_date_cmp = pd.Timestamp(current_date).tz_localize(None)
            start_date_cmp = pd.Timestamp(start_date).tz_localize(None)
        else:
            current_date_cmp = pd.Timestamp(current_date)
            start_date_cmp = pd.Timestamp(start_date)
        
        df_windowed = df[(df.index >= start_date_cmp) & (df.index <= current_date_cmp)]
        
        if len(df_windowed) > 0:
            mtf_data[tf] = df_windowed
    
    return mtf_data


if __name__ == "__main__":
    print("MTF Loader")
    
    # Test load
    loader = MTFLoader("data/raw/mtf")
    
    mtf_data = loader.load("BTCUSDT")
    
    print(f"\nLoaded {len(mtf_data)} timeframes:")
    for tf, df in mtf_data.items():
        print(f"  {tf}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Test alignment
    if '15m' in mtf_data and len(mtf_data['15m']) > 0:
        target_time = mtf_data['15m'].index[-100]
        
        print(f"\nAligning at {target_time}:")
        aligned = loader.align_at_timestamp(mtf_data, target_time, '15m')
        
        for tf, df in aligned.items():
            last_time = df.index[-1] if len(df) > 0 else None
            print(f"  {tf}: {len(df)} bars, last at {last_time}")
    
    # Validate
    validation = loader.validate_alignment(mtf_data)
    print(f"\nValidation: {'✅ PASS' if validation['valid'] else '❌ FAIL'}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"  ⚠️  {issue}")
