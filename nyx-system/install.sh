#!/bin/bash

# NYX Trading System - Installation Script
# One-click setup for development environment

set -e  # Exit on error

echo ""
echo "████████████████████████████████████████████████████████████████"
echo "NYX TRADING SYSTEM v0.8 - INSTALLATION"
echo "████████████████████████████████████████████████████████████████"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "➤ Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    echo -e "${RED}✗ Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
    else
        echo "Using existing virtual environment"
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "➤ Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "➤ Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "➤ Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "➤ Installing Python dependencies..."
echo "  (This may take a few minutes...)"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Install package in development mode
echo ""
echo "➤ Installing NYX package..."
pip install -e .
echo -e "${GREEN}✓ NYX package installed${NC}"

# Create data directories
echo ""
echo "➤ Setting up data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/results
mkdir -p data/sample
mkdir -p logs
echo -e "${GREEN}✓ Data directories ready${NC}"

# Check for sample data
echo ""
echo "➤ Checking for sample data..."
if [ -f "data/sample/BTCUSDT_1h_sample.csv" ]; then
    echo -e "${GREEN}✓ Sample data found${NC}"
else
    echo -e "${YELLOW}⚠ No sample data found${NC}"
    echo "  You can add sample CSV files to data/sample/"
fi

# Dashboard setup
echo ""
echo "➤ Dashboard setup..."
if command -v npm &> /dev/null; then
    echo "  npm detected"
    read -p "Install dashboard dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd dashboard
        echo "  Installing Node packages..."
        npm install
        cd ..
        echo -e "${GREEN}✓ Dashboard dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping dashboard setup${NC}"
    fi
else
    echo -e "${YELLOW}⚠ npm not found. Skipping dashboard setup.${NC}"
    echo "  Install Node.js to use the dashboard"
fi

# Run tests
echo ""
echo "➤ Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short 2>&1 | tail -20
    echo -e "${GREEN}✓ Tests completed${NC}"
else
    echo -e "${YELLOW}⚠ pytest not installed. Skipping tests.${NC}"
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ INSTALLATION COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download data:"
echo "   python scripts/download_data.py --pair BTCUSDT --since 2020-01-01"
echo ""
echo "3. Run backtest:"
echo "   python scripts/run_backtest.py --pair BTCUSDT"
echo ""
echo "4. Launch dashboard:"
echo "   # Terminal 1:"
echo "   cd dashboard && uvicorn api:app --reload --port 8000"
echo ""
echo "   # Terminal 2:"
echo "   cd dashboard && npm start"
echo ""
echo "For help:"
echo "   python scripts/download_data.py --help"
echo "   python scripts/run_backtest.py --help"
echo ""
echo "════════════════════════════════════════════════════════════════"
