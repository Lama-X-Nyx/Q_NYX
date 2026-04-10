"""
Live Feed - Binance Public API

FREE - No API keys needed
Rate limit: 1200 requests/minute

Usage:
    feed = BinanceFeed()
    price = feed.get_price('BTCUSDT')
    candles = feed.get_recent_candles('BTCUSDT', limit=100)
"""

import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime
import time


class BinanceFeed:
    """
    Binance Public API Wrapper
    
    Free endpoints (no authentication):
    - Current prices
    - Recent candles (OHLCV)
    - Market data
    
    Rate limit: 1200 requests/minute
    """
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self, pairs: Optional[List[str]] = None):
        """
        Initialize Binance feed
        
        Args:
            pairs: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        """
        self.pairs = pairs or ['BTCUSDT', 'ETHUSDT']
        self.session = requests.Session()
        
        print("📡 Binance Feed Initialized")
        print(f"   Pairs: {', '.join(self.pairs)}")
        print(f"   API: {self.BASE_URL} (public, free)")
    
    def get_price(self, pair: str) -> Optional[float]:
        """
        Get current price for pair
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
        
        Returns:
            Current price or None if error
        """
        try:
            url = f"{self.BASE_URL}/api/v3/ticker/price"
            params = {'symbol': pair}
            
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
        
        except Exception as e:
            print(f"⚠️ Error fetching price for {pair}: {e}")
            return None
    
    def get_recent_candles(
        self, 
        pair: str, 
        interval: str = '1h', 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get recent OHLCV candles
        
        Args:
            pair: Trading pair
            interval: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles (max 1000)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            url = f"{self.BASE_URL}/api/v3/klines"
            params = {
                'symbol': pair,
                'interval': interval,
                'limit': min(limit, 1000)
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Keep only relevant columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            print(f"⚠️ Error fetching candles for {pair}: {e}")
            return None
    
    def get_all_prices(self) -> dict:
        """
        Get prices for all configured pairs
        
        Returns:
            {pair: price}
        """
        prices = {}
        
        for pair in self.pairs:
            price = self.get_price(pair)
            if price:
                prices[pair] = price
            
            # Rate limit protection
            time.sleep(0.1)
        
        return prices
    
    def get_ticker_24h(self, pair: str) -> Optional[dict]:
        """
        Get 24h ticker statistics
        
        Args:
            pair: Trading pair
        
        Returns:
            {
                'price': current price,
                'change_24h': % change,
                'high_24h': 24h high,
                'low_24h': 24h low,
                'volume_24h': 24h volume
            }
        """
        try:
            url = f"{self.BASE_URL}/api/v3/ticker/24hr"
            params = {'symbol': pair}
            
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'volume_24h': float(data['volume'])
            }
        
        except Exception as e:
            print(f"⚠️ Error fetching 24h ticker for {pair}: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """
        Check if Binance market is operational
        
        Returns:
            True if market is open
        """
        try:
            url = f"{self.BASE_URL}/api/v3/ping"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        
        except:
            return False
    
    def get_server_time(self) -> Optional[datetime]:
        """
        Get Binance server time
        
        Returns:
            Server datetime
        """
        try:
            url = f"{self.BASE_URL}/api/v3/time"
            response = self.session.get(url, timeout=5)
            data = response.json()
            
            timestamp = data['serverTime'] / 1000
            return datetime.fromtimestamp(timestamp)
        
        except Exception as e:
            print(f"⚠️ Error fetching server time: {e}")
            return None


if __name__ == "__main__":
    # Test
    print("="*60)
    print("BINANCE FEED - Test")
    print("="*60)
    
    feed = BinanceFeed(pairs=['BTCUSDT', 'ETHUSDT'])
    
    # Test market open
    print(f"\nMarket open: {feed.is_market_open()}")
    
    # Test server time
    server_time = feed.get_server_time()
    print(f"Server time: {server_time}")
    
    # Test price
    btc_price = feed.get_price('BTCUSDT')
    print(f"\nBTC Price: ${btc_price:,.2f}")
    
    # Test 24h ticker
    ticker = feed.get_ticker_24h('BTCUSDT')
    if ticker:
        print(f"24h Change: {ticker['change_24h']:+.2f}%")
        print(f"24h High: ${ticker['high_24h']:,.2f}")
        print(f"24h Low: ${ticker['low_24h']:,.2f}")
    
    # Test candles
    print(f"\nFetching recent candles...")
    candles = feed.get_recent_candles('BTCUSDT', interval='1h', limit=10)
    if candles is not None:
        print(f"✓ Got {len(candles)} candles")
        print(candles.tail())
    
    # Test all prices
    print(f"\nAll prices:")
    prices = feed.get_all_prices()
    for pair, price in prices.items():
        print(f"  {pair}: ${price:,.2f}")
    
    print("\n✓ Test complete")
