## 🌟 Support This Project

Found this project useful? Please consider:

- ⭐ **Starring this repository** - It helps others discover the project
- 🍴 **Forking** and contributing improvements  
- 📢 **Sharing** with the trading and quantitative finance community
- 💡 **Opening issues** for bugs or feature requests
- 🚀 **Contributing** code, documentation, or examples

**Created with ❤️ by [Kristofer Meio-Renn](https://github.com/kmrlab)**

---

## How To Share

Spread the word about this project! Share with the trading and quantitative finance community:

- 📘 [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://github.com/kmrlab/algo-allocation)
- 💼 [Share on LinkedIn](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/kmrlab/algo-allocation)
- 📱 [Share on Telegram](https://t.me/share/url?url=https://github.com/kmrlab/algo-allocation&text=Professional-grade%20portfolio%20allocation%20algorithms%20for%20crypto%20trading%20strategies)
- 🐦 [Share on X (Twitter)](https://twitter.com/intent/tweet?url=https://github.com/kmrlab/algo-allocation&text=Check%20out%20these%20portfolio%20allocation%20algorithms%20for%20crypto%20trading%20strategies!%20%23QuantitativeFinance%20%23CryptoTrading%20%23OpenSource)

---

# Portfolio Allocation Algorithms

[![Stars](https://img.shields.io/github/stars/kmrlab/algo-allocation?style=social)](https://github.com/kmrlab/algo-allocation)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)

Portfolio allocation algorithms for crypto trading strategies

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [File Descriptions](#-file-descriptions)
  - [allocation_by_efficiency.py](#allocation_by_efficiencypy)
  - [allocation_cvar.py](#allocation_cvarpy)
  - [allocation_ivw.py](#allocation_ivwpy)
  - [allocation_performance_based.py](#allocation_performance_basedpy)
  - [data.csv](#datacsv)
  - [requirements.txt](#requirementstxt)
- [Usage Examples](#-usage-examples)
- [Supported Strategies](#-supported-strategies)
- [Mathematical Background](#-mathematical-background)
- [Visualization](#-visualization)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

## 🎯 Overview

This repository contains a suite of portfolio allocation algorithms designed specifically for cryptocurrency trading strategies and assets. The implementations focus on risk management, return optimization, and portfolio diversification using quantitative finance techniques.

The project provides multiple allocation methodologies suitable for different risk profiles and market conditions, from conservative risk-parity approaches to aggressive return-maximizing strategies.

## ✨ Features

- **Multiple Allocation Methods**: 4+ different allocation algorithms
- **Risk Management**: CVaR optimization and drawdown constraints
- **Performance Analysis**: Comprehensive metrics calculation
- **Visualization**: Professional charts and reports
- **Real Data**: Historical performance data for 5 trading strategies
- **Production Ready**: Clean, documented code with error handling
- **Configurable**: Adjustable parameters and constraints
- **Export Functionality**: CSV output for further analysis

## 🚀 Installation

### Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn for visualization

### Quick Start

```bash
# Clone the repository
git clone https://github.com/kmrlab/algo-allocation.git
cd algo-allocation

# Install dependencies
pip install -r requirements.txt

# Run an example
python allocation_by_efficiency.py
```

## 📁 Project Structure

```
algo-allocation/
├── allocation_by_efficiency.py    # Multi-method portfolio allocator
├── allocation_cvar.py            # CVaR optimization
├── allocation_ivw.py             # Inverse Volatility Weighting
├── allocation_performance_based.py # Performance-based allocation
├── data.csv                      # Historical strategy returns
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📋 File Descriptions

### `allocation_by_efficiency.py`

**Advanced Multi-Method Portfolio Allocator**

A comprehensive portfolio allocation system that combines four different allocation methodologies:

- **Risk-Based**: Inverse risk weighting (1/drawdown)
- **Return-Weighted**: Proportional to expected returns
- **Efficiency-Based**: Return/risk ratio optimization
- **Market Cap Weighted**: Traditional market capitalization approach

**Key Features:**
- Configurable method weights (e.g., 25% risk, 25% return, 35% efficiency, 15% cap)
- Weight constraints (default: 4%-18% per asset)
- Comprehensive reporting with metrics
- Visual comparison of all methods
- Supports both crypto assets and trading strategies

**Use Case**: Ideal for balanced portfolio construction with customizable risk/return preferences.

### `allocation_cvar.py`

**Conditional Value at Risk (CVaR) Optimization**

Professional risk management approach using CVaR optimization for trading strategy allocation:

- **CVaR Calculation**: 95% confidence level tail risk measurement
- **Constrained Optimization**: Minimum return targets with risk constraints
- **Strategy Analysis**: Performance metrics for 5 trading strategies
- **Weight Bounds**: 5%-40% allocation limits per strategy

**Key Features:**
- SLSQP optimization algorithm
- Annual return and volatility calculations
- Correlation analysis between strategies
- Professional visualization suite
- Export results for further analysis

**Use Case**: Perfect for institutional-grade risk management and regulatory compliance.

### `allocation_ivw.py`

**Inverse Volatility Weighting (IVW)**

Classic risk-parity approach that allocates capital inversely proportional to asset volatility:

- **Formula**: wᵢ = (1/σᵢ) / Σⱼ(1/σⱼ)
- **Risk Focus**: Lower volatility = higher allocation
- **Performance Metrics**: Sharpe ratio, drawdown, total return
- **5 Trading Strategies**: Trend, Mean-reversion, Pair-trading, Market-making, Arbitrage

**Key Features:**
- Automatic volatility calculation from historical returns
- Risk minimization objective
- Clear methodology explanation
- Professional dark-theme visualizations

**Use Case**: Conservative approach for risk-averse investors seeking stable returns.

### `allocation_performance_based.py`

**Multi-Factor Performance-Based Allocation**

Sophisticated allocation method combining multiple performance metrics:

- **Return Factor**: Total historical returns (50% weight)
- **Sharpe Factor**: Risk-adjusted returns (30% weight)  
- **Drawdown Factor**: Maximum drawdown protection (20% weight)
- **Normalization**: [0,1] scaling for fair comparison

**Key Features:**
- Configurable factor weights
- Normalized scoring system
- Combined performance score
- Handles negative drawdowns appropriately

**Use Case**: Performance-oriented allocation for growth-focused portfolios.

### `data.csv`

**Historical Strategy Performance Data**

Complete dataset containing daily returns for 5 trading strategies over 366 days:

- **Date Range**: January 1, 2024 - December 31, 2024
- **Strategies**: Trend, Mean-reversion, Pair-Trading, Market-Making, Arbitrage
- **Format**: Clean CSV with proper date parsing
- **Data Quality**: No missing values, consistent formatting

**Statistics:**
- 366 trading days of data
- 5 different strategy types
- Daily return format (decimal percentages)
- Ready for immediate analysis

### `requirements.txt`

**Python Dependencies Specification**

Carefully curated dependency list with version constraints:

- **Core Scientific**: NumPy (≥1.24.0), Pandas (≥2.0.0), SciPy (≥1.10.0)
- **Visualization**: Matplotlib (≥3.7.0), Seaborn (≥0.12.0)
- **Version Compatibility**: Tested with Python 3.8+
- **Development Tools**: Optional dependencies for enhanced development

**Features:**
- Stable version ranges
- Virtual environment compatible
- Clear installation instructions
- Optional development dependencies

## 💻 Usage Examples

### Basic Portfolio Allocation

```python
from allocation_by_efficiency import SimplePortfolioAllocator

# Your asset data
crypto_data = [
    {'name': 'BTC', 'return': 24.24, 'drawdown': 1.27, 'cap': 2090440108182},
    {'name': 'ETH', 'return': 42.03, 'drawdown': 16.72, 'cap': 290734266974},
    # ... more assets
]

# Create allocator
allocator = SimplePortfolioAllocator(
    data=crypto_data,
    min_weight=0.04,  # 4% minimum
    max_weight=0.18   # 18% maximum
)

# Define method weights
method_weights = {
    'risk_based': 0.25,
    'return_weighted': 0.25, 
    'efficiency': 0.35,
    'cap_weighted': 0.15
}

# Generate allocation and report
methods, final, constrained = allocator.calculate_final_allocation(method_weights)
report, results_table = allocator.generate_report(method_weights)
print(report)
```

### CVaR Optimization

```python
from allocation_cvar import main

# Run CVaR optimization with historical data
results = main()  # Automatically loads data.csv

# Results include:
# - Optimal strategy weights
# - Portfolio metrics (return, volatility, Sharpe ratio)
# - Risk measures (VaR, CVaR)
# - Visualization outputs
```

### Inverse Volatility Weighting

```python
from allocation_ivw import inverse_volatility_weighting

# Calculate IVW allocation
weights, allocations, metrics, volatilities = inverse_volatility_weighting(
    historical_returns=HISTORICAL_RETURNS, 
    total_capital=100000
)

# View results
print(f"Weights: {weights}")
print(f"Allocations: {allocations}")
```

## 🎯 Supported Strategies

The algorithms support allocation across 5 different trading strategies:

1. **Trend Following**: Momentum-based directional trading
2. **Mean Reversion**: Statistical arbitrage and reversion strategies  
3. **Pair Trading**: Market-neutral relative value strategies
4. **Market Making**: Liquidity provision and spread capture
5. **Arbitrage**: Risk-free profit from price discrepancies

Each strategy has distinct risk-return characteristics suitable for portfolio diversification.

## 📊 Mathematical Background

### Efficiency Calculation
```
Efficiency = Expected Return / Maximum Drawdown
```

### CVaR Formula
```
CVaR_α = E[X | X ≤ VaR_α]
where α is the confidence level (typically 95%)
```

### Inverse Volatility Weighting
```
w_i = (1/σ_i) / Σ_j(1/σ_j)
where σ_i is the volatility of asset i
```

### Performance Score
```
Score = 0.5 × Return_norm + 0.3 × Sharpe_norm + 0.2 × (-Drawdown_norm)
```

## 📈 Visualization

All algorithms include professional visualization capabilities:

- **Portfolio Allocation**: Pie charts and bar graphs
- **Risk-Return Scatter**: Strategy positioning analysis  
- **Performance Comparison**: Side-by-side method comparison
- **Correlation Heatmaps**: Inter-strategy relationships
- **Cumulative Returns**: Historical performance tracking

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Additional allocation algorithms (Black-Litterman, Risk Budgeting, etc.)
- Enhanced visualization and reporting
- Performance optimization and backtesting
- Documentation and examples
- Bug fixes and code quality improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**IMPORTANT RISK DISCLOSURE**

This software is provided for educational and research purposes only. The algorithms and methodologies contained herein are not investment advice and should not be construed as such.

### Key Risk Factors:

- **Past Performance**: Historical data does not guarantee future results
- **Market Risk**: Cryptocurrency and trading strategies are subject to high volatility
- **Model Risk**: Mathematical models may not accurately predict market behavior  
- **Implementation Risk**: Code may contain bugs or errors
- **Regulatory Risk**: Trading regulations vary by jurisdiction

### Professional Disclaimer:

- **No Investment Advice**: This is not personalized investment advice
- **Due Diligence Required**: Conduct your own research before making investment decisions
- **Professional Consultation**: Consult with qualified financial advisors
- **Risk Management**: Never invest more than you can afford to lose
- **Backtesting Limitations**: Simulated performance may not reflect actual trading results

### Liability Limitation:

The authors and contributors shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use of this software, including but not limited to financial losses, trading losses, or missed opportunities.

**USE AT YOUR OWN RISK**

By using this software, you acknowledge that you understand these risks and agree to use the software solely at your own discretion and risk.

---

**Happy Trading! 📈**
