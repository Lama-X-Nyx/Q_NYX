# NYX Dashboard

React + FastAPI dashboard for NYX Trading System

## 🚀 Quick Start

### Backend (FastAPI)

```bash
# Install dependencies
pip install fastapi uvicorn

# Run API server
cd dashboard
uvicorn api:app --reload --port 8000

# API will be available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend (React)

```bash
# Install Node dependencies
cd dashboard
npm install

# Start development server
npm start

# Dashboard will open at http://localhost:3000
```

## 📡 API Endpoints

### Core Endpoints

```
GET  /                      - API info
GET  /api/health           - Health check
GET  /api/pairs            - List available pairs
GET  /api/backtest/{pair}  - Run backtest on pair
POST /api/backtest         - Run backtest with parameters
GET  /api/results/{pair}   - Get saved results
GET  /api/live/price/{pair} - Get live price
GET  /api/data/{pair}      - Get historical data
POST /api/optimize         - Run parameter optimization
GET  /api/stats/summary    - Summary stats for all pairs
```

### Example Requests

```bash
# Get available pairs
curl http://localhost:8000/api/pairs

# Run backtest
curl http://localhost:8000/api/backtest/BTCUSDT

# Get live price
curl http://localhost:8000/api/live/price/ETHUSDT

# Run backtest with parameters
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{"pair": "BTCUSDT", "start_date": "2020-01-01", "capital": 10000}'
```

## 🎨 Dashboard Features

- **Real-time Price Tracking** - Live price updates every 5 seconds
- **One-Click Backtesting** - Run backtests directly from UI
- **Performance Metrics** - View Return, CAGR, Win Rate, etc.
- **Trade History** - Recent trades with P&L
- **Multi-Pair Support** - Switch between different trading pairs
- **Responsive Design** - Works on desktop and mobile

## 🔧 Configuration

### Environment Variables

Create `.env` file in dashboard directory:

```env
REACT_APP_API_URL=http://localhost:8000
```

### CORS

For production, update CORS settings in `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📦 Production Build

### Build React App

```bash
cd dashboard
npm run build

# Serve with static file server
npx serve -s build -p 3000
```

### Deploy API

```bash
# Using uvicorn
uvicorn dashboard.api:app --host 0.0.0.0 --port 8000

# Or with gunicorn
gunicorn dashboard.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional)

```dockerfile
# Dockerfile for API
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing

### Test API

```bash
# Health check
curl http://localhost:8000/api/health

# List pairs
curl http://localhost:8000/api/pairs

# Run backtest
curl http://localhost:8000/api/backtest/BTCUSDT
```

### Test Frontend

```bash
# Run React tests
cd dashboard
npm test
```

## 📊 Adding New Features

### Add New API Endpoint

```python
# In dashboard/api.py
@app.get("/api/custom/{param}")
async def custom_endpoint(param: str):
    return {"param": param, "result": "success"}
```

### Add New React Component

```javascript
// In dashboard/src/components/NewComponent.jsx
import React from 'react';

const NewComponent = () => {
  return <div>New Component</div>;
};

export default NewComponent;
```

## 🐛 Troubleshooting

### CORS Errors

If you see CORS errors in browser console:
1. Make sure API is running on port 8000
2. Check CORS middleware in `api.py`
3. Verify `proxy` in `package.json`

### Connection Refused

If frontend can't connect to API:
1. Verify API is running: `curl http://localhost:8000/api/health`
2. Check firewall settings
3. Ensure correct API_URL in `.env`

### Data Not Found

If you see "Data not found" errors:
1. Make sure you've downloaded data: `python scripts/download_data.py --all`
2. Check data exists in `data/raw/` directory
3. Verify file naming: `{PAIR}_{TIMEFRAME}.csv`

## 📝 Notes

- API uses simplified backtest engine for speed
- Live prices are simulated (replace with real Binance WebSocket in production)
- Results are saved to `data/results/` directory
- Dashboard uses Tailwind CSS for styling

## 🔗 Links

- FastAPI Docs: http://localhost:8000/docs
- React App: http://localhost:3000
- GitHub: [Your Repo URL]
