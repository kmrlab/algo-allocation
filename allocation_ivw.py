"""
██╗  ██╗███╗   ███╗██████╗ ██╗      █████╗ ██████╗ 
██║ ██╔╝████╗ ████║██╔══██╗██║     ██╔══██╗██╔══██╗
█████╔╝ ██╔████╔██║██████╔╝██║     ███████║██████╔╝
██╔═██╗ ██║╚██╔╝██║██╔══██╗██║     ██╔══██║██╔══██╗
██║  ██╗██║ ╚═╝ ██║██║  ██║███████╗██║  ██║██████╔╝
╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ 

Crafted with ❤️ by Kristofer Meio-Renn

Found this useful? Star the repo to show your support! Thank you!
GitHub: https://github.com/kmrlab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================
TOTAL_CAPITAL = 100000  # USDT
STRATEGIES = ['Trend', 'Mean-reversion', 'Pair-Trading', 'Market-Making', 'Arbitrage']

# Historical performance data (last 30 days) - SAME DATA
HISTORICAL_RETURNS = {
    'Trend': [0.02, -0.01, 0.03, 0.01, -0.005, 0.025, -0.01, 0.02, 0.01, -0.005,
              0.015, 0.02, -0.008, 0.01, 0.005, 0.03, -0.012, 0.018, 0.008, 0.002,
              0.025, -0.01, 0.015, 0.005, -0.003, 0.02, 0.01, -0.005, 0.012, 0.008],
    'Mean-reversion': [0.008, 0.012, -0.01, 0.015, 0.005, -0.008, 0.018, 0.003, 0.01, 0.007,
                       -0.005, 0.02, 0.008, -0.003, 0.012, 0.005, 0.015, -0.01, 0.008, 0.01,
                       0.003, 0.018, -0.005, 0.012, 0.008, -0.002, 0.015, 0.005, 0.01, 0.007],
    'Pair-Trading': [0.005, 0.008, 0.012, -0.003, 0.01, 0.006, -0.005, 0.015, 0.002, 0.008,
                     0.01, -0.002, 0.012, 0.005, 0.008, -0.003, 0.015, 0.006, 0.01, -0.001,
                     0.008, 0.003, 0.012, -0.002, 0.01, 0.005, 0.008, 0.015, -0.003, 0.006],
    'Market-Making': [0.003, 0.005, 0.004, 0.008, 0.002, 0.006, 0.007, 0.001, 0.009, 0.004,
                      0.005, 0.003, 0.008, 0.006, 0.002, 0.007, 0.004, 0.005, 0.003, 0.008,
                      0.001, 0.009, 0.005, 0.006, 0.004, 0.007, 0.003, 0.008, 0.002, 0.005],
    'Arbitrage': [0.001, 0.003, 0.002, 0.004, 0.001, 0.005, 0.002, 0.003, 0.001, 0.004,
                  0.002, 0.001, 0.005, 0.003, 0.002, 0.004, 0.001, 0.003, 0.002, 0.005,
                  0.001, 0.004, 0.002, 0.003, 0.001, 0.005, 0.002, 0.004, 0.001, 0.003]
}

# Visualization settings
plt.style.use('dark_background')
COLORS = ['#00d4aa', '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']

# Minimum weight to prevent division by zero
MIN_VOLATILITY = 1e-6

# =============================================================================
# MAIN CODE
# =============================================================================

def calculate_performance_metrics(returns):
    """Calculate strategy performance metrics"""
    returns_array = np.array(returns)
    
    # Total return
    total_return = np.prod(1 + returns_array) - 1
    
    # Average daily return
    mean_daily_return = np.mean(returns_array)
    
    # Volatility (standard deviation)
    volatility = np.std(returns_array)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = mean_daily_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns / running_max - 1
    max_drawdown = np.min(drawdowns)
    
    return {
        'total_return': total_return,
        'mean_daily_return': mean_daily_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def inverse_volatility_weighting(historical_returns, total_capital):
    """Inverse Volatility Weighting (IVW) method"""
    
    # Calculate metrics for each strategy
    metrics = {}
    volatilities = []
    
    for strategy, returns in historical_returns.items():
        metrics[strategy] = calculate_performance_metrics(returns)
        volatilities.append(max(metrics[strategy]['volatility'], MIN_VOLATILITY))
    
    # Calculate inverse volatilities
    inverse_volatilities = np.array([1/vol for vol in volatilities])
    
    # Normalization to get weights: wᵢ = (1/σᵢ) / Σⱼ(1/σⱼ)
    weights = inverse_volatilities / inverse_volatilities.sum()
    
    # Capital allocation
    allocations = weights * total_capital
    
    return weights, allocations, metrics, volatilities


def create_visualization(weights, allocations, metrics, volatilities, method_name):
    """Create visualization of results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{method_name}\nAllocation of $100,000 USDT', fontsize=16, fontweight='bold')
    
    # 1. Pie chart - distribution in %
    ax1.pie(weights, labels=STRATEGIES, autopct='%1.1f%%', colors=COLORS, startangle=90)
    ax1.set_title('Weight Distribution (%)', fontweight='bold')
    
    # 2. Bar chart - distribution in $
    bars = ax2.bar(STRATEGIES, allocations, color=COLORS)
    ax2.set_title('Capital Allocation (USDT)', fontweight='bold')
    ax2.set_ylabel('Capital (USDT)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add values on bars
    for bar, allocation in zip(bars, allocations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${allocation:,.0f}', ha='center', va='bottom')
    
    # 3. Strategy volatility
    volatility_percent = [vol * 100 for vol in volatilities]
    bars3 = ax3.bar(STRATEGIES, volatility_percent, color=COLORS)
    ax3.set_title('Strategy Volatility (%)', fontweight='bold')
    ax3.set_ylabel('Volatility (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, vol in zip(bars3, volatility_percent):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{vol:.2f}%', ha='center', va='bottom')
    
    # 4. Inverse volatility (basis for weights)
    inverse_vols = [1/vol for vol in volatilities]
    bars4 = ax4.bar(STRATEGIES, inverse_vols, color=COLORS)
    ax4.set_title('Inverse Volatility (1/σ)', fontweight='bold')
    ax4.set_ylabel('1/σ')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, inv_vol in zip(bars4, inverse_vols):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{inv_vol:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def print_detailed_results(weights, allocations, metrics, volatilities):
    """Output detailed results"""
    
    print("=" * 80)
    print("INVERSE VOLATILITY WEIGHTING (IVW) - DETAILED RESULTS")
    print("=" * 80)
    
    # Create DataFrame for nice output
    results_data = []
    inverse_volatilities = [1/vol for vol in volatilities]
    
    for i, strategy in enumerate(STRATEGIES):
        results_data.append({
            'Strategy': strategy,
            'Volatility (%)': f"{volatilities[i]*100:.2f}%",
            '1/σ': f"{inverse_volatilities[i]:.2f}",
            'Weight (%)': f"{weights[i]*100:.1f}%",
            'Capital (USDT)': f"${allocations[i]:,.0f}",
            'Total Return (%)': f"{metrics[strategy]['total_return']*100:.2f}%",
            'Sharpe Ratio': f"{metrics[strategy]['sharpe_ratio']:.3f}"
        })
    
    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))
    
    print(f"\nTotal Capital: ${sum(allocations):,.0f} USDT")
    print(f"Weight Sum Check: {sum(weights):.3f} (should be 1.000)")
    print(f"Sum of Inverse Volatilities: {sum(inverse_volatilities):.2f}")
    
    print("\nIVW Principle:")
    print("- Strategies with LOWER volatility get HIGHER weight")
    print("- Formula: wᵢ = (1/σᵢ) / Σⱼ(1/σⱼ)")
    print("- Goal: reduce overall portfolio risk")

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Starting Inverse Volatility Weighting (IVW)...")
    
    # Calculate allocation
    weights, allocations, metrics, volatilities = inverse_volatility_weighting(HISTORICAL_RETURNS, TOTAL_CAPITAL)
    
    # Output results
    print_detailed_results(weights, allocations, metrics, volatilities)
    
    # Create visualization
    create_visualization(weights, allocations, metrics, volatilities, "Inverse Volatility Weighting (IVW)")
    
    print("\nAnalysis completed!")