# NYX Trading System v0.8

Professional algorithmic trading system combining Semi-Markov HMM, Smart Money Concepts, and advanced risk management.

## 🎯 Features

- **Semi-Markov Hidden Markov Model (HSMM)** - State detection with duration modeling
- **Smart Money Concepts (SMC)** - Order Blocks, Fair Value Gaps
- **Regime Detection** - Bull/Bear/Range market classification
- **Position Pyramiding** - Dynamic position scaling
- **Monte Carlo Validation** - Robust performance testing
- **Multi-Asset Support** - BTC, ETH, SOL, and more
- **Real-time Dashboard** - React-based interactive UI

## 📊 Performance

| Asset | Return | CAGR | Win Rate | Sharpe |
|-------|--------|------|----------|--------|
| BTC   | +34.41% | 7.10% | 34.5% | 0.89 |
| ETH   | +58.86% | 12.00% | 36.2% | 0.91 |
| SOL   | TBD | TBD | TBD | TBD |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nyx-system.git
cd nyx-system

# Install dependencies
pip install -r requirements.txt

# Configure
cp config/config.yaml.example config/config.yaml
# Edit config.yaml with your settings
```

### Download Data

```bash
# Download historical data for backtesting
python scripts/download_data.py --pair BTCUSDT --timeframes 15m,1h,4h --since 2020-01-01
```

### Run Backtest

```bash
# Single pair backtest
python scripts/run_backtest.py --pair ETHUSDT

# Multi-pair backtest
python scripts/run_backtest.py --all

# With parameter optimization
python scripts/optimize.py --pair BTCUSDT
```

### Launch Dashboard

```bash
# Start backend API
uvicorn dashboard.api:app --reload

# Start frontend (separate terminal)
cd dashboard
npm install
npm start

# Dashboard opens at http://localhost:3000
```

## 📁 Project Structure

```
nyx-system/
├── config/          # Configuration files
├── src/
│   ├── core/        # Core components (HSMM, SMC, Indicators)
│   ├── strategy/    # Trading strategies
│   ├── data/        # Data management
│   ├── backtest/    # Backtesting engine
│   ├── risk/        # Risk management
│   └── execution/   # Order execution
├── dashboard/       # React dashboard
├── tests/           # Unit tests
├── scripts/         # Utility scripts
└── data/            # Historical data
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- **Strategy parameters** (SdC threshold, pyramiding, etc.)
- **Risk settings** (position sizing, stop loss)
- **Data sources** (Binance, local files)
- **Execution** (live/paper trading)

See `config/pairs.yaml` for pair-specific settings.

## 📈 Usage Examples

### Basic Backtest

```python
from src.backtest.engine import BacktestEngine
from src.strategy.nyx_v05 import NYXStrategy

# Initialize
engine = BacktestEngine(config='config/config.yaml')
strategy = NYXStrategy()

# Run backtest
results = engine.run(
    strategy=strategy,
    pair='ETHUSDT',
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Print results
print(f"Return: {results.total_return:.2f}%")
print(f"CAGR: {results.cagr:.2f}%")
print(f"Win Rate: {results.win_rate:.1f}%")
```

### Live Trading (Paper)

```python
from src.execution.broker import BinanceBroker
from src.strategy.nyx_v05 import NYXStrategy

# Initialize
broker = BinanceBroker(paper_trading=True)
strategy = NYXStrategy()

# Start trading
strategy.run(broker=broker, pairs=['ETHUSDT'])
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_hsmm.py
```

## 📊 Dashboard

The React dashboard provides:

- **Real-time monitoring** - Live prices, positions, P&L
- **Performance analytics** - Equity curve, drawdown, metrics
- **Parameter controls** - Backtest with different settings
- **Trade execution** - Manual trading interface
- **Multi-asset view** - Portfolio allocation

## 🔐 Security

- **Never commit credentials** - Use `.env` or `credentials.yaml` (gitignored)
- **API keys** - Store in `config/credentials.yaml`
- **Production** - Use environment variables

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Discord**: NYX Trading Community

## ⚠️ Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly before live trading.

---

**Built with ❤️ by the NYX Team**
