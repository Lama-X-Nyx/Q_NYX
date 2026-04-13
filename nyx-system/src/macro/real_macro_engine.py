"""
Real Macro Engine v0.7 - Event-based with temporal decay

Improvements over v0.6:
- Real macro events from JSON database
- Temporal decay (exponential)
- Asset-specific sensitivity
- Event aggregation with weights
- No hindsight bias (events evaluated at publication time)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class RealMacroEngine:
    """
    Real macro engine based on historical events
    
    Features:
    - Event database (JSON)
    - Temporal decay (exponential)
    - Asset-specific sensitivity
    - Event aggregation
    - No hindsight bias
    
    Usage:
        engine = RealMacroEngine('data/macro_events.json')
        signal = engine.get_macro_signal('BTC', '2022-06-01')
    """
    
    def __init__(self, events_file: str = 'data/macro_events.json'):
        """
        Initialize macro engine
        
        Args:
            events_file: Path to macro events JSON
        """
        self.events_file = events_file
        self.events = self._load_events()
        
        # Default asset sensitivities (if not in event)
        self.default_sensitivity = {
            'BTC': {
                'FED_RATE_CUT': 0.9,
                'FED_RATE_HIKE': 0.85,
                'FED_QE': 1.0,
                'BTC_HALVING': 1.0,
                'BTC_ETF': 1.0,
                'BANK_CRISIS': 0.7
            },
            'ETH': {
                'FED_RATE_CUT': 0.85,
                'FED_RATE_HIKE': 0.9,
                'FED_QE': 0.95,
                'BTC_HALVING': 0.3,
                'BTC_ETF': 0.3,
                'ETH_UPGRADE': 1.0
            }
        }
    
    def _load_events(self) -> List[Dict]:
        """Load events from JSON file"""
        
        if not Path(self.events_file).exists():
            print(f"⚠ Events file not found: {self.events_file}")
            return []
        
        with open(self.events_file, 'r') as f:
            data = json.load(f)
        
        events = data.get('events', [])
        
        # Parse dates
        for event in events:
            event['date'] = pd.to_datetime(event['date'])
        
        return events
    
    def get_macro_signal(self, asset: str, current_date: "str | pd.Timestamp") -> Dict:
        """
        Get macro signal for asset at specific date

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            current_date: Date string 'YYYY-MM-DD' or Timestamp

        Returns:
            {
                'signal': 'BULLISH' or 'BEARISH' or 'NEUTRAL',
                'strength': 0-1,
                'active_events': [...],
                'cumulative_impact': float
            }
        """

        current_ts = pd.to_datetime(current_date)

        # Get active events (within decay window)
        active_events = self._get_active_events(asset, current_ts)

        if not active_events:
            return {
                'signal': 'NEUTRAL',
                'strength': 0.0,
                'active_events': [],
                'cumulative_impact': 0.0
            }

        # Calculate cumulative impact
        cumulative_impact = self._calculate_cumulative_impact(
            active_events,
            asset,
            current_ts
        )
        
        # Determine signal
        if cumulative_impact > 0.3:
            signal = 'BULLISH'
        elif cumulative_impact < -0.3:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        strength = abs(cumulative_impact)
        
        return {
            'signal': signal,
            'strength': min(strength, 1.0),
            'active_events': active_events,
            'cumulative_impact': cumulative_impact
        }
    
    def _get_active_events(self, asset: str, current_date: pd.Timestamp) -> List[Dict]:
        """
        Get events that are still active (within decay window)
        
        Args:
            asset: Asset symbol
            current_date: Current timestamp
        
        Returns:
            List of active events
        """
        active = []
        
        for event in self.events:
            event_date = event['date']
            
            # Only consider events in the past
            if event_date > current_date:
                continue
            
            # Check if asset is affected
            if asset not in event.get('assets_affected', []):
                continue
            
            # Check if still within decay window
            decay_days = event.get('decay_days', 60)
            days_elapsed = (current_date - event_date).days
            
            if days_elapsed <= decay_days:
                active.append({
                    'event': event,
                    'days_elapsed': days_elapsed,
                    'decay_days': decay_days
                })
        
        return active
    
    def _calculate_cumulative_impact(
        self, 
        active_events: List[Dict], 
        asset: str, 
        current_date: pd.Timestamp
    ) -> float:
        """
        Calculate cumulative impact from all active events
        
        Uses exponential decay: impact = base_impact * exp(-lambda * t)
        
        Args:
            active_events: List of active events
            asset: Asset symbol
            current_date: Current date
        
        Returns:
            Cumulative impact (-1 to +1)
        """
        
        total_impact = 0.0
        
        for active in active_events:
            event = active['event']
            days_elapsed = active['days_elapsed']
            decay_days = active['decay_days']
            
            # Base impact
            impact_type = event['impact']
            base_strength = event.get('strength', 1.0)
            
            # Asset-specific sensitivity
            asset_sensitivity = event.get('asset_sensitivity', {}).get(asset, 1.0)
            
            # Handle MIXED impact (like SVB crisis)
            if impact_type == 'MIXED':
                # Initially bearish, becomes bullish
                if days_elapsed < decay_days * 0.3:
                    base_impact = -0.5  # Bearish first 30%
                else:
                    base_impact = 0.5   # Bullish after
            else:
                base_impact = 1.0 if impact_type == 'BULLISH' else -1.0
            
            # Exponential decay
            # lambda = 5 / decay_days (95% decay at decay_days)
            decay_lambda = 5.0 / decay_days
            decay_factor = np.exp(-decay_lambda * days_elapsed)
            
            # Final impact for this event
            event_impact = (
                base_impact * 
                base_strength * 
                asset_sensitivity * 
                decay_factor
            )
            
            total_impact += event_impact
        
        # Normalize to [-1, 1]
        return np.tanh(total_impact)
    
    def get_event_details(self, current_date: "str | pd.Timestamp") -> List[Dict]:
        """
        Get detailed view of recent events

        Args:
            current_date: Date string or Timestamp

        Returns:
            List of recent events with details
        """
        current_ts = pd.to_datetime(current_date)
        
        recent_events = []
        
        for event in self.events:
            if event['date'] <= current_ts:
                days_ago = (current_ts - event['date']).days
                
                if days_ago <= 180:  # Last 6 months
                    recent_events.append({
                        'date': event['date'].strftime('%Y-%m-%d'),
                        'type': event['type'],
                        'title': event['title'],
                        'impact': event['impact'],
                        'days_ago': days_ago,
                        'still_active': days_ago <= event.get('decay_days', 60)
                    })
        
        # Sort by date (most recent first)
        recent_events.sort(key=lambda x: x['days_ago'])
        
        return recent_events
    
    def add_event(
        self, 
        date: str, 
        event_type: str, 
        title: str,
        impact: str,
        strength: float = 1.0,
        decay_days: int = 60,
        assets_affected: Optional[List[str]] = None,
        asset_sensitivity: Optional[Dict] = None
    ):
        """
        Add a new macro event
        
        Args:
            date: Event date 'YYYY-MM-DD'
            event_type: Type (FED_RATE_CUT, etc.)
            title: Event title
            impact: BULLISH, BEARISH, or NEUTRAL
            strength: 0-1
            decay_days: Days until full decay
            assets_affected: List of assets
            asset_sensitivity: Dict of asset sensitivities
        """
        
        new_event = {
            'date': date,
            'type': event_type,
            'title': title,
            'impact': impact,
            'strength': strength,
            'decay_days': decay_days,
            'assets_affected': assets_affected or ['BTC', 'ETH'],
            'asset_sensitivity': asset_sensitivity or {}
        }
        
        self.events.append(new_event)
        
        # Save to file
        self._save_events()
    
    def _save_events(self):
        """Save events back to JSON file"""
        
        # Convert dates back to strings
        events_to_save = []
        for event in self.events:
            event_copy = event.copy()
            if isinstance(event_copy['date'], pd.Timestamp):
                event_copy['date'] = event_copy['date'].strftime('%Y-%m-%d')
            events_to_save.append(event_copy)
        
        data = {
            'events': events_to_save,
            'metadata': {
                'version': '1.0',
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Historical macro events'
            }
        }
        
        with open(self.events_file, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("REAL MACRO ENGINE v0.7 - Example Usage")
    print("="*80)
    
    # Initialize
    engine = RealMacroEngine('data/macro_events.json')
    
    print(f"\n✓ Loaded {len(engine.events)} macro events")
    
    # Test dates
    test_dates = [
        '2020-04-01',  # After COVID QE
        '2022-07-01',  # During rate hikes
        '2023-11-01',  # ETF speculation
        '2024-05-01'   # After BTC halving
    ]
    
    for date in test_dates:
        print(f"\n{'='*80}")
        print(f"Date: {date}")
        print(f"{'='*80}")
        
        for asset in ['BTC', 'ETH']:
            signal = engine.get_macro_signal(asset, date)
            
            print(f"\n{asset}:")
            print(f"  Signal: {signal['signal']}")
            print(f"  Strength: {signal['strength']:.2f}")
            print(f"  Impact: {signal['cumulative_impact']:+.2f}")
            print(f"  Active Events: {len(signal['active_events'])}")
            
            if signal['active_events']:
                print(f"\n  Recent Events:")
                for active in signal['active_events'][:3]:
                    event = active['event']
                    print(f"    • {event['title']} ({active['days_elapsed']}d ago)")
    
    print("\n" + "="*80)
    print("✅ Real Macro Engine Working")
    print("="*80)
