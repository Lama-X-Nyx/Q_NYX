#!/usr/bin/env python3
"""
NYX Trading System - Setup Verification
Checks that everything is installed correctly
"""

import sys
import os
from pathlib import Path

def color_text(text, color):
    """Add color to terminal output"""
    colors = {
        'green': '\033[0;32m',
        'yellow': '\033[1;33m',
        'red': '\033[0;31m',
        'blue': '\033[0;34m',
        'reset': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def check_python_version():
    """Check Python version"""
    print(f"\n{'='*60}")
    print("PYTHON VERSION")
    print(f"{'='*60}")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print(color_text(f"✓ Python {version_str}", 'green'))
        return True
    else:
        print(color_text(f"✗ Python {version_str} (3.8+ required)", 'red'))
        return False

def check_dependencies():
    """Check required Python packages"""
    print(f"\n{'='*60}")
    print("PYTHON DEPENDENCIES")
    print(f"{'='*60}")
    
    required = [
        'numpy', 'pandas', 'scipy', 'sklearn',
        'fastapi', 'uvicorn', 'pyyaml', 'pytest'
    ]
    
    installed = []
    missing = []
    
    for package in required:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            installed.append(package)
            print(color_text(f"✓ {package}", 'green'))
        except ImportError:
            missing.append(package)
            print(color_text(f"✗ {package}", 'red'))
    
    return len(missing) == 0

def check_modules():
    """Check NYX modules can be imported"""
    print(f"\n{'='*60}")
    print("NYX MODULES")
    print(f"{'='*60}")
    
    modules = [
        ('src.core.hsmm', 'SemiMarkovHMM'),
        ('src.core.smc', 'SMCDetector'),
    ]
    
    success = True
    
    for module_path, class_name in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(color_text(f"✓ {module_path}.{class_name}", 'green'))
        except Exception as e:
            print(color_text(f"✗ {module_path}.{class_name}: {e}", 'red'))
            success = False
    
    return success

def check_scripts():
    """Check CLI scripts are available"""
    print(f"\n{'='*60}")
    print("CLI SCRIPTS")
    print(f"{'='*60}")
    
    scripts = [
        'scripts/download_data.py',
        'scripts/run_backtest.py',
        'scripts/optimize.py'
    ]
    
    success = True
    
    for script in scripts:
        if os.path.exists(script):
            print(color_text(f"✓ {script}", 'green'))
        else:
            print(color_text(f"✗ {script}", 'red'))
            success = False
    
    return success

def check_data_directories():
    """Check data directories exist"""
    print(f"\n{'='*60}")
    print("DATA DIRECTORIES")
    print(f"{'='*60}")
    
    directories = [
        'data/raw',
        'data/processed',
        'data/results',
        'data/sample',
        'logs'
    ]
    
    success = True
    
    for directory in directories:
        if os.path.exists(directory):
            # Count files
            files = list(Path(directory).glob('*.csv'))
            count = len(files)
            print(color_text(f"✓ {directory} ({count} files)", 'green'))
        else:
            print(color_text(f"✗ {directory}", 'red'))
            success = False
    
    return success

def check_configuration():
    """Check configuration files"""
    print(f"\n{'='*60}")
    print("CONFIGURATION FILES")
    print(f"{'='*60}")
    
    configs = [
        'config/config.yaml',
        'config/pairs.yaml'
    ]
    
    success = True
    
    for config in configs:
        if os.path.exists(config):
            size = os.path.getsize(config)
            print(color_text(f"✓ {config} ({size} bytes)", 'green'))
        else:
            print(color_text(f"✗ {config}", 'red'))
            success = False
    
    return success

def check_dashboard():
    """Check dashboard files"""
    print(f"\n{'='*60}")
    print("DASHBOARD")
    print(f"{'='*60}")
    
    files = [
        'dashboard/api.py',
        'dashboard/App.jsx',
        'dashboard/package.json'
    ]
    
    success = True
    
    for file in files:
        if os.path.exists(file):
            print(color_text(f"✓ {file}", 'green'))
        else:
            print(color_text(f"✗ {file}", 'red'))
            success = False
    
    # Check if node_modules exists
    if os.path.exists('dashboard/node_modules'):
        print(color_text(f"✓ dashboard/node_modules", 'green'))
    else:
        print(color_text(f"⚠ dashboard/node_modules (run: cd dashboard && npm install)", 'yellow'))
    
    return success

def check_tests():
    """Check test files"""
    print(f"\n{'='*60}")
    print("TESTS")
    print(f"{'='*60}")
    
    tests = [
        'tests/test_hsmm.py',
        'tests/test_smc.py',
        'tests/test_api.py',
        'tests/test_backtest.py'
    ]
    
    success = True
    
    for test in tests:
        if os.path.exists(test):
            # Count test functions
            with open(test, 'r') as f:
                content = f.read()
                test_count = content.count('def test_')
            print(color_text(f"✓ {test} ({test_count} tests)", 'green'))
        else:
            print(color_text(f"✗ {test}", 'red'))
            success = False
    
    return success

def main():
    """Run all checks"""
    print("\n" + "█"*60)
    print("NYX TRADING SYSTEM - SETUP VERIFICATION")
    print("█"*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("NYX Modules", check_modules),
        ("CLI Scripts", check_scripts),
        ("Data Directories", check_data_directories),
        ("Configuration", check_configuration),
        ("Dashboard", check_dashboard),
        ("Tests", check_tests)
    ]
    
    results = {}
    
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        color = 'green' if status else 'red'
        print(color_text(f"{symbol} {name}", color))
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print(color_text("\n✅ SYSTEM READY!", 'green'))
        print("\nYou can now:")
        print("  • Run backtests: python scripts/run_backtest.py --help")
        print("  • Launch dashboard: uvicorn dashboard.api:app --reload")
        print("  • Run tests: pytest tests/ -v")
        return 0
    else:
        print(color_text("\n⚠ SOME CHECKS FAILED", 'yellow'))
        print("\nRun: ./install.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
