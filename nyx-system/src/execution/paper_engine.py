"""
Paper Trading Engine v0.8

Features:
- SQLite logging (market snapshots, signals, trades)
- Position management (1 position max per pair)
- Stop loss / Take profit
- PnL tracking
- No pyramiding (v0.8 simplification)

Usage:
    engine = PaperEngine(initial_capital=10000)
    engine.execute_signal(pair, signal, current_price)
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


class PaperEngine:
    """
    Paper trading execution engine
    
    Rules v0.8:
    - 1 position max per pair
    - No pyramiding
    - Stop loss / Take profit enforcement
    - Full SQLite logging
    """
    
    def __init__(self, initial_capital: float = 10000, db_path: str = 'data/paper_trading.db'):
        """
        Initialize paper engine
        
        Args:
            initial_capital: Starting capital (USD)
            db_path: Path to SQLite database
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.db_path = db_path
        
        # Positions: {pair: Position}
        self.positions = {}
        
        # Initialize database
        self.db = sqlite3.connect(db_path)
        self._init_db()
        
        print(f"💰 Paper Engine Initialized")
        print(f"   Capital: ${self.capital:,.2f}")
        print(f"   Database: {db_path}")
    
    def _init_db(self):
        """Create SQLite tables"""
        
        # Market snapshots
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                price REAL NOT NULL,
                hsmm_state TEXT,
                confidence REAL,
                regime TEXT,
                macro_signal TEXT,
                macro_strength REAL
            )
        ''')
        
        # Signals
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                position_size REAL,
                stop_loss REAL,
                take_profit REAL,
                reasons TEXT,
                FOREIGN KEY(snapshot_id) REFERENCES market_snapshots(id)
            )
        ''')
        
        # Trades
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                pair TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                FOREIGN KEY(signal_id) REFERENCES signals(id)
            )
        ''')
        
        # Equity curve
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                capital REAL NOT NULL,
                equity REAL NOT NULL,
                num_positions INTEGER
            )
        ''')
        
        self.db.commit()
        
        print("   ✓ Database initialized")
    
    def execute_signal(self, pair: str, signal: Dict, current_price: float) -> Optional[str]:
        """
        Execute signal from NYXEngine
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            signal: Signal dict from NYXEngine
            current_price: Current market price
        
        Returns:
            Action taken ('ENTER', 'EXIT', 'HOLD', 'STOP', 'TP')
        """
        
        # Log snapshot
        snapshot_id = self._log_snapshot(pair, signal, current_price) or 0

        # Log signal
        signal_id = self._log_signal(snapshot_id, signal) or 0

        # Check existing position first
        if pair in self.positions:
            # Manage position
            action = self._manage_position(pair, signal, current_price, signal_id)
            return action

        # Check entry
        if signal['action'] == 'BUY' and pair not in self.positions:
            self._enter_position(pair, signal, current_price, signal_id)
            return 'ENTER'
        
        return 'HOLD'
    
    def _log_snapshot(self, pair: str, signal: Dict, price: float) -> Optional[int]:
        """Log market snapshot"""
        
        cursor = self.db.execute('''
            INSERT INTO market_snapshots 
            (timestamp, pair, price, hsmm_state, confidence, regime, macro_signal, macro_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            pair,
            price,
            signal.get('hsmm_state'),
            signal.get('confidence'),
            signal.get('regime'),
            signal.get('macro_signal'),
            signal.get('macro_strength')
        ))
        
        self.db.commit()
        return cursor.lastrowid
    
    def _log_signal(self, snapshot_id: int, signal: Dict) -> Optional[int]:
        """Log trading signal"""
        
        cursor = self.db.execute('''
            INSERT INTO signals
            (snapshot_id, action, position_size, stop_loss, take_profit, reasons)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            snapshot_id,
            signal['action'],
            signal.get('position_size'),
            signal.get('stop_loss'),
            signal.get('take_profit'),
            json.dumps(signal.get('reasons', []))
        ))
        
        self.db.commit()
        return cursor.lastrowid
    
    def _enter_position(self, pair: str, signal: Dict, price: float, signal_id: int):
        """Enter new position"""
        
        # Calculate quantity
        position_value = self.capital * signal['position_size']
        quantity = position_value / price
        
        # Deduct from capital
        self.capital -= position_value
        
        # Create position
        self.positions[pair] = {
            'entry_price': price,
            'entry_time': datetime.now(),
            'quantity': quantity,
            'position_value': position_value,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'signal_id': signal_id
        }
        
        # Log trade (entry)
        self.db.execute('''
            INSERT INTO trades
            (signal_id, pair, entry_time, entry_price, quantity)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            signal_id,
            pair,
            datetime.now().isoformat(),
            price,
            quantity
        ))
        
        self.db.commit()
        
        print(f"\n  🟢 ENTER LONG {pair}")
        print(f"     Price: ${price:,.2f}")
        print(f"     Quantity: {quantity:.6f}")
        print(f"     Value: ${position_value:,.2f}")
        print(f"     SL: ${signal['stop_loss']:.2f}")
        print(f"     TP: ${signal['take_profit']:.2f}")
    
    def _manage_position(self, pair: str, signal: Dict, price: float, signal_id: int) -> str:
        """Manage existing position"""
        
        position = self.positions[pair]
        
        # Check stop loss
        if price <= position['stop_loss']:
            self._close_position(pair, price, "Stop Loss", signal_id)
            return 'STOP'
        
        # Check take profit
        if price >= position['take_profit']:
            self._close_position(pair, price, "Take Profit", signal_id)
            return 'TP'
        
        # Check exit signal
        if signal['action'] == 'SELL':
            reason = signal['reasons'][0] if signal['reasons'] else 'Exit Signal'
            self._close_position(pair, price, reason, signal_id)
            return 'EXIT'
        
        return 'HOLD'
    
    def _close_position(self, pair: str, price: float, reason: str, signal_id: int):
        """Close position"""
        
        position = self.positions[pair]
        
        # Calculate PnL
        exit_value = position['quantity'] * price
        pnl = exit_value - position['position_value']
        pnl_pct = (pnl / position['position_value']) * 100
        
        # Update capital
        self.capital += exit_value
        
        # Update trade in database
        self.db.execute('''
            UPDATE trades
            SET exit_time = ?,
                exit_price = ?,
                pnl = ?,
                pnl_pct = ?,
                exit_reason = ?
            WHERE pair = ? 
            AND exit_time IS NULL
        ''', (
            datetime.now().isoformat(),
            price,
            pnl,
            pnl_pct,
            reason,
            pair
        ))
        
        self.db.commit()
        
        print(f"\n  🔴 CLOSE {pair}")
        print(f"     Exit Price: ${price:,.2f}")
        print(f"     PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"     Reason: {reason}")
        
        # Remove position
        del self.positions[pair]
    
    def update_equity(self) -> float:
        """
        Calculate and log current equity
        
        Returns:
            Total equity (capital + open positions value)
        """
        
        total_equity = self.capital
        
        # Add value of open positions (would need current prices)
        # For now, just use entry value as approximation
        for pair, pos in self.positions.items():
            total_equity += pos['position_value']
        
        # Log to database
        self.db.execute('''
            INSERT INTO equity_history
            (timestamp, capital, equity, num_positions)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            self.capital,
            total_equity,
            len(self.positions)
        ))
        
        self.db.commit()
        
        return total_equity
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        
        # Get all closed trades
        cursor = self.db.execute('''
            SELECT pnl, pnl_pct, exit_reason
            FROM trades
            WHERE exit_time IS NOT NULL
        ''')
        
        trades = cursor.fetchall()
        
        if not trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        total_trades = len(trades)
        winners = [t for t in trades if t[0] > 0]
        losers = [t for t in trades if t[0] <= 0]
        
        win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t[0] for t in trades)
        total_return = (total_pnl / self.initial_capital) * 100
        
        avg_win = (sum(t[0] for t in winners) / len(winners)) if winners else 0
        avg_loss = (sum(t[0] for t in losers) / len(losers)) if losers else 0
        
        return {
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'open_positions': len(self.positions)
        }
    
    def print_status(self):
        """Print current status"""
        
        equity = self.update_equity()
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print(f"📊 PAPER TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print(f"Capital:        ${self.capital:,.2f}")
        print(f"Total Equity:   ${equity:,.2f}")
        print(f"Total Return:   {stats['total_return']:+.2f}%")
        print(f"\nTrades:         {stats['total_trades']}")
        print(f"Win Rate:       {stats['win_rate']:.1f}%")
        print(f"Open Positions: {stats['open_positions']}")
        
        if self.positions:
            print("\nOpen Positions:")
            for pair, pos in self.positions.items():
                print(f"  {pair}:")
                print(f"    Entry: ${pos['entry_price']:.2f}")
                print(f"    Quantity: {pos['quantity']:.6f}")
                print(f"    Value: ${pos['position_value']:,.2f}")
        
        print("="*60)
    
    def close(self):
        """Close database connection"""
        self.db.close()


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("PAPER ENGINE v0.8 - Test")
    print("="*60)
    
    # Create engine
    engine = PaperEngine(initial_capital=10000)
    
    # Simulate signal
    test_signal = {
        'action': 'BUY',
        'confidence': 7.5,
        'position_size': 0.10,  # 10%
        'stop_loss': 67000,
        'take_profit': 72000,
        'regime': 'Bull',
        'macro_signal': 'BULLISH',
        'macro_strength': 0.8,
        'reasons': ['Test entry'],
        'hsmm_state': 'Trend+'
    }
    
    # Execute
    action = engine.execute_signal('BTCUSDT', test_signal, 68000)
    print(f"\nAction: {action}")
    
    # Status
    engine.print_status()
    
    # Close
    engine.close()
    print("\n✓ Test complete")
