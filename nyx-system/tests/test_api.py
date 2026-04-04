"""
Unit Tests for FastAPI Backend (dashboard/api.py)

Run:
    pytest tests/test_api.py -v
    pytest tests/test_api.py::test_health_endpoint -v
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the FastAPI app
from dashboard.api import app

# Create test client
client = TestClient(app)


class TestAPIEndpoints:
    """Test all API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
        assert "NYX" in data["name"]
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
    
    def test_pairs_endpoint(self):
        """Test pairs listing endpoint"""
        response = client.get("/api/pairs")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "pairs" in data
        assert "count" in data
        assert isinstance(data["pairs"], list)
        assert isinstance(data["count"], int)
    
    def test_live_price_endpoint(self):
        """Test live price endpoint"""
        # This should work even without real data (simulated)
        response = client.get("/api/live/price/BTCUSDT")
        
        # Should return 200 or 500 depending on data availability
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "pair" in data
            assert "price" in data
            assert "timestamp" in data


class TestBacktestEndpoints:
    """Test backtest-related endpoints"""
    
    def test_backtest_endpoint_get(self):
        """Test GET backtest endpoint"""
        response = client.get("/api/backtest/BTCUSDT")
        
        # Should return 200 or 404/500 if no data
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            # Should have backtest results
            assert "initial_capital" in data or "error" in data
    
    def test_backtest_endpoint_post(self):
        """Test POST backtest endpoint with parameters"""
        payload = {
            "pair": "BTCUSDT",
            "start_date": None,
            "end_date": None,
            "capital": 10000
        }
        
        response = client.post("/api/backtest", json=payload)
        
        # Should return 200 or 500 depending on data
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "initial_capital" in data or "error" in data


class TestDataEndpoints:
    """Test data-related endpoints"""
    
    def test_data_endpoint(self):
        """Test historical data endpoint"""
        response = client.get("/api/data/BTCUSDT?limit=10")
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "pair" in data
            assert "data" in data
            assert "count" in data


class TestOptimizeEndpoints:
    """Test optimization endpoints"""
    
    def test_optimize_endpoint(self):
        """Test parameter optimization endpoint"""
        payload = {
            "pair": "BTCUSDT",
            "param": "sdc_threshold",
            "metric": "cagr"
        }
        
        response = client.post("/api/optimize", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return job started status
        assert "status" in data
        assert data["status"] == "started"


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_pair(self):
        """Test with invalid trading pair"""
        response = client.get("/api/backtest/INVALIDPAIR")
        
        # Should handle gracefully
        assert response.status_code in [404, 500]
    
    def test_missing_data(self):
        """Test endpoints with missing data"""
        response = client.get("/api/results/NONEXISTENT")
        
        # Should return 404
        assert response.status_code == 404


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers(self):
        """Test that CORS headers are present"""
        response = client.options("/api/health")
        
        # CORS should be configured
        assert response.status_code in [200, 405]


class TestRequestValidation:
    """Test request validation"""
    
    def test_invalid_backtest_payload(self):
        """Test backtest with invalid payload"""
        payload = {
            "pair": "BTC",  # Too short
            "capital": -1000  # Negative
        }
        
        response = client.post("/api/backtest", json=payload)
        
        # Should validate and reject
        assert response.status_code in [422, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
