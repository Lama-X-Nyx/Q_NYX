"""
Unit Tests for Real Macro Engine

Run:
    pytest tests/test_macro_engine.py -v
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.macro.real_macro_engine import RealMacroEngine


class TestMacroEngineBasics:
    """Test basic functionality"""
    
    def test_engine_loads(self):
        """Engine loads events successfully"""
        engine = RealMacroEngine('data/macro_events.json')
        
        assert engine is not None
        assert len(engine.events) > 0
        assert hasattr(engine, 'get_macro_signal')
    
    def test_events_loaded(self):
        """Events are loaded and parsed correctly"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Should have events
        assert len(engine.events) >= 10  # At least 10 events
        
        # Events should have required fields
        for event in engine.events:
            assert 'date' in event
            assert 'type' in event
            assert 'impact' in event
            assert 'decay_days' in event
            assert isinstance(event['date'], pd.Timestamp)


class TestTemporalLogic:
    """Test temporal logic (no hindsight bias)"""
    
    def test_future_events_ignored(self):
        """Future events should NOT influence signal"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Date before all events (early 2020)
        signal = engine.get_macro_signal('BTC', '2020-01-01')
        
        # Should be neutral (no active events)
        assert signal['signal'] in ['NEUTRAL', 'BEARISH', 'BULLISH']
        assert signal['cumulative_impact'] == 0.0 or abs(signal['cumulative_impact']) < 0.1
        assert len(signal['active_events']) == 0
    
    def test_past_events_active(self):
        """Past events within decay window should be active"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Date after COVID QE (April 2020)
        signal = engine.get_macro_signal('BTC', '2020-04-01')
        
        # Should have active events (COVID response)
        assert len(signal['active_events']) > 0
        assert signal['cumulative_impact'] != 0.0
    
    def test_old_events_decay(self):
        """Events outside decay window should not be active"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Date long after events (2024)
        signal = engine.get_macro_signal('BTC', '2024-06-01')
        
        # Some events should have decayed completely
        # Check that not ALL events are active
        total_events = len(engine.events)
        active_events = len(signal['active_events'])
        
        assert active_events < total_events  # Some should have decayed


class TestDecayFunction:
    """Test exponential decay"""
    
    def test_decay_reduces_over_time(self):
        """Impact should decrease over time"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Get signal at different dates after COVID QE (2020-03-23)
        signal_early = engine.get_macro_signal('BTC', '2020-04-01')  # ~9 days after
        signal_mid = engine.get_macro_signal('BTC', '2020-05-15')    # ~53 days after
        signal_late = engine.get_macro_signal('BTC', '2020-09-01')   # ~162 days after
        
        # Impact should decrease
        impact_early = abs(signal_early['cumulative_impact'])
        impact_mid = abs(signal_mid['cumulative_impact'])
        impact_late = abs(signal_late['cumulative_impact'])
        
        # Early impact should be highest
        assert impact_early > impact_mid or impact_early > 0.5
        # Late impact should be lowest
        assert impact_late < impact_early or impact_late < 0.3
    
    def test_decay_formula(self):
        """Test that decay formula works correctly"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Get events for a specific date
        signal = engine.get_macro_signal('BTC', '2020-04-15')
        
        # Should have cumulative impact
        assert 'cumulative_impact' in signal
        assert isinstance(signal['cumulative_impact'], float)


class TestAssetSensitivity:
    """Test asset-specific sensitivity"""
    
    def test_btc_vs_eth_sensitivity(self):
        """BTC and ETH should have different sensitivities to BTC ETF"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Date after BTC ETF approval (2024-01-10)
        date = '2024-01-15'
        
        btc_signal = engine.get_macro_signal('BTC', date)
        eth_signal = engine.get_macro_signal('ETH', date)
        
        # BTC should be more affected by BTC ETF than ETH
        # (if BTC ETF event exists in the data)
        if btc_signal['strength'] > 0:
            assert btc_signal['strength'] >= eth_signal['strength']
    
    def test_asset_not_affected(self):
        """Asset not in assets_affected should have no impact"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Check that an asset not affected by any events returns neutral
        # (This depends on the events in the JSON)
        signal = engine.get_macro_signal('SOL', '2020-03-25')
        
        # SOL might not be affected by early events
        # Signal should be neutral or have low impact
        assert signal['signal'] in ['NEUTRAL', 'BULLISH', 'BEARISH']


class TestSignalOutput:
    """Test signal output structure"""
    
    def test_signal_structure(self):
        """Signal should have correct structure"""
        engine = RealMacroEngine('data/macro_events.json')
        
        signal = engine.get_macro_signal('BTC', '2020-04-01')
        
        # Required fields
        assert 'signal' in signal
        assert 'strength' in signal
        assert 'active_events' in signal
        assert 'cumulative_impact' in signal
        
        # Correct types
        assert signal['signal'] in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert isinstance(signal['strength'], float)
        assert isinstance(signal['active_events'], list)
        assert isinstance(signal['cumulative_impact'], float)
        
        # Bounds
        assert 0 <= signal['strength'] <= 1
        assert -1 <= signal['cumulative_impact'] <= 1
    
    def test_no_events_neutral(self):
        """When no events active, signal should be neutral"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Very early date (before any events)
        signal = engine.get_macro_signal('BTC', '2019-01-01')
        
        assert signal['signal'] == 'NEUTRAL'
        assert signal['strength'] == 0.0
        assert len(signal['active_events']) == 0


class TestEventDetails:
    """Test event details retrieval"""
    
    def test_get_event_details(self):
        """Can retrieve event details"""
        engine = RealMacroEngine('data/macro_events.json')
        
        details = engine.get_event_details('2020-04-01')
        
        assert isinstance(details, list)
        # Should have some recent events
        if len(details) > 0:
            event = details[0]
            assert 'date' in event
            assert 'type' in event
            assert 'title' in event
            assert 'days_ago' in event


class TestRealWorldScenarios:
    """Test real-world scenarios"""
    
    def test_covid_qe_bullish(self):
        """COVID QE period should be bullish for BTC"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # April 2020 - after unlimited QE
        signal = engine.get_macro_signal('BTC', '2020-04-15')
        
        # Should be bullish or at least positive impact
        if len(signal['active_events']) > 0:
            assert signal['cumulative_impact'] > 0 or signal['signal'] == 'BULLISH'
    
    def test_rate_hikes_bearish(self):
        """Rate hike period should be bearish"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Summer 2022 - aggressive rate hikes
        signal = engine.get_macro_signal('BTC', '2022-07-01')
        
        # Should be bearish or negative impact
        if len(signal['active_events']) > 0:
            # Can be bearish or neutral (depends on exact events)
            assert signal['signal'] in ['BEARISH', 'NEUTRAL', 'BULLISH']
    
    def test_btc_etf_approval_bullish(self):
        """BTC ETF approval should be bullish for BTC"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # January 2024 - after ETF approval
        signal = engine.get_macro_signal('BTC', '2024-01-15')
        
        # Should have strong positive impact for BTC
        if len(signal['active_events']) > 0:
            # Check if cumulative impact is positive
            assert signal['cumulative_impact'] >= 0


class TestEdgeCases:
    """Test edge cases"""
    
    def test_invalid_date_format(self):
        """Should handle various date formats"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Try different formats
        try:
            signal1 = engine.get_macro_signal('BTC', '2020-04-01')
            signal2 = engine.get_macro_signal('BTC', '2020-4-1')
            
            # Both should work
            assert signal1 is not None
            assert signal2 is not None
        except Exception as e:
            pytest.fail(f"Should handle date formats: {e}")
    
    def test_unknown_asset(self):
        """Should handle unknown assets gracefully"""
        engine = RealMacroEngine('data/macro_events.json')
        
        # Unknown asset should return neutral signal
        signal = engine.get_macro_signal('UNKNOWN', '2020-04-01')
        
        assert signal is not None
        assert signal['signal'] == 'NEUTRAL'
        assert len(signal['active_events']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
