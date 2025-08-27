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

# Historical performance data (last 30 days)
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

# Weight coefficients for Performance-based allocation
RETURN_WEIGHT = 0.5
SHARPE_WEIGHT = 0.3
MAX_DD_WEIGHT = 0.2

# Visualization settings
plt.style.use('dark_background')
COLORS = ['#00d4aa', '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']

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

def performance_based_allocation(historical_returns, total_capital):
    """Performance-based allocation method"""
    
    # Calculate metrics for each strategy
    metrics = {}
    for strategy, returns in historical_returns.items():
        metrics[strategy] = calculate_performance_metrics(returns)
    
    # Normalize metrics for weight calculation
    returns_scores = []
    sharpe_scores = []
    dd_scores = []
    
    strategy_names = list(metrics.keys())
    
    for strategy in strategy_names:
        returns_scores.append(metrics[strategy]['total_return'])
        sharpe_scores.append(metrics[strategy]['sharpe_ratio'])
        # For max drawdown: lower drawdown is better (invert)
        dd_scores.append(-metrics[strategy]['max_drawdown'])
    
    # Normalize to range [0, 1]
    returns_scores = np.array(returns_scores)
    sharpe_scores = np.array(sharpe_scores)
    dd_scores = np.array(dd_scores)
    
    returns_normalized = (returns_scores - returns_scores.min()) / (returns_scores.max() - returns_scores.min())
    sharpe_normalized = (sharpe_scores - sharpe_scores.min()) / (sharpe_scores.max() - sharpe_scores.min())
    dd_normalized = (dd_scores - dd_scores.min()) / (dd_scores.max() - dd_scores.min())
    
    # Combined score
    combined_scores = (RETURN_WEIGHT * returns_normalized + 
                      SHARPE_WEIGHT * sharpe_normalized + 
                      MAX_DD_WEIGHT * dd_normalized)
    
    # Convert to weights
    weights = combined_scores / combined_scores.sum()
    
    # Capital allocation
    allocations = weights * total_capital
    
    return weights, allocations, metrics

def create_visualization(weights, allocations, metrics, method_name):
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
    
    # 3. Performance metrics - Total Return
    total_returns = [metrics[strategy]['total_return'] * 100 for strategy in STRATEGIES]
    bars3 = ax3.bar(STRATEGIES, total_returns, color=COLORS)
    ax3.set_title('Total Return (%)', fontweight='bold')
    ax3.set_ylabel('Return (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, ret in zip(bars3, total_returns):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ret:.1f}%', ha='center', va='bottom')
    
    # 4. Sharpe Ratio
    sharpe_ratios = [metrics[strategy]['sharpe_ratio'] for strategy in STRATEGIES]
    bars4 = ax4.bar(STRATEGIES, sharpe_ratios, color=COLORS)
    ax4.set_title('Sharpe Ratio', fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, sharpe in zip(bars4, sharpe_ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{sharpe:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def print_detailed_results(weights, allocations, metrics):
    """Output detailed results"""
    
    print("=" * 80)
    print("PERFORMANCE-BASED ALLOCATION - DETAILED RESULTS")
    print("=" * 80)
    
    # Create DataFrame for nice output
    results_data = []
    for i, strategy in enumerate(STRATEGIES):
        results_data.append({
            'Strategy': strategy,
            'Weight (%)': f"{weights[i]*100:.1f}%",
            'Capital (USDT)': f"${allocations[i]:,.0f}",
            'Total Return (%)': f"{metrics[strategy]['total_return']*100:.2f}%",
            'Sharpe Ratio': f"{metrics[strategy]['sharpe_ratio']:.3f}",
            'Max Drawdown (%)': f"{metrics[strategy]['max_drawdown']*100:.2f}%",
            'Volatility (%)': f"{metrics[strategy]['volatility']*100:.2f}%"
        })
    
    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))
    
    print(f"\nTotal Capital: ${sum(allocations):,.0f} USDT")
    print(f"Weight Sum Check: {sum(weights):.3f} (should be 1.000)")
    
    print("\nMethod weight coefficients:")
    print(f"- Return: {RETURN_WEIGHT*100}%")
    print(f"- Sharpe Ratio: {SHARPE_WEIGHT*100}%") 
    print(f"- Max Drawdown: {MAX_DD_WEIGHT*100}%")

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Starting Performance-Based Allocation...")
    
    # Calculate allocation
    weights, allocations, metrics = performance_based_allocation(HISTORICAL_RETURNS, TOTAL_CAPITAL)
    
    # Output results
    print_detailed_results(weights, allocations, metrics)
    
    # Create visualization
    create_visualization(weights, allocations, metrics, "Performance-Based Allocation")
    
    print("\nAnalysis completed!")