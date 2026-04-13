from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nyx-trading",
    version="0.2.5",
    author="NYX Team",
    author_email="team@nyx-trading.com",
    description="Professional algorithmic trading system with HSMM and SMC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nyx-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "ccxt>=4.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0"],
        "dashboard": ["streamlit>=1.24.0", "plotly>=5.14.0"],
    },
    entry_points={
        "console_scripts": [
            "nyx-backtest=scripts.run_backtest:main",
            "nyx-download=scripts.download_data:main",
            "nyx-optimize=scripts.optimize:main",
        ],
    },
)
