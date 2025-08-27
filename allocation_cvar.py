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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# ==================== SETTINGS ====================
TOTAL_CAPITAL = 1_000_000  # Total fund capital
CONFIDENCE_LEVEL = 0.95    # Confidence level for CVaR (95%)
TARGET_RETURN = 0.15       # Minimum target return (15% annually)

# Fund strategies
STRATEGIES = ['Trend', 'Mean-reversion', 'Pair-Trading', 'Market-Making', 'Arbitrage']

# Visualization style
plt.style.use('seaborn-v0_8')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50']
sns.set_palette(COLORS)

# ==================== DATA LOADING ====================

def load_strategy_data_from_csv(csv_file='data.csv'):
    """Load historical data from CSV file"""
    try:
        print(f"📂 Loading data from file '{csv_file}'...")
        data = pd.read_csv(csv_file, parse_dates=['Date'], index_col='Date')
        
        # Check for all required strategies
        missing_strategies = set(STRATEGIES) - set(data.columns)
        if missing_strategies:
            print(f"❌ Missing strategies in CSV file: {missing_strategies}")
            return None
            
        # Select only required strategies in correct order
        data = data[STRATEGIES]
        
        print(f"✅ Loaded {len(data)} days of data")
        print(f"📅 Period: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}")
        
        return data
        
    except FileNotFoundError:
        print(f"⚠️ File '{csv_file}' not found")
        return None
    except Exception as e:
        print(f"❌ Error loading file '{csv_file}': {e}")
        return None



# ==================== CVaR OPTIMIZATION ====================

def calculate_cvar(returns, weights, alpha=0.05):
    """Calculate Conditional Value at Risk (CVaR)"""
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, alpha * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return -cvar  # Return positive value for minimization

def calculate_var(returns, weights, alpha=0.05):
    """Calculate Value at Risk (VaR)"""
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, alpha * 100)
    return -var

def portfolio_expected_return(returns, weights):
    """Calculate portfolio expected return"""
    return np.mean(returns @ weights)

def cvar_optimization(returns, target_return=TARGET_RETURN/366):
    """CVaR optimization with constraints"""
    n_assets = len(returns.columns)
    
    # Initial weights (equal-weighted portfolio)
    x0 = np.array([1/n_assets] * n_assets)
    
    # Constraints
    constraints = [
        # Sum of weights = 1
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        # Minimum expected return
        {'type': 'ineq', 'fun': lambda x: portfolio_expected_return(returns, x) - target_return}
    ]
    
    # Weight bounds (minimum 5%, maximum 40%)
    bounds = [(0.05, 0.40) for _ in range(n_assets)]
    
    # Optimization
    result = minimize(
        fun=lambda x: calculate_cvar(returns, x, alpha=1-CONFIDENCE_LEVEL),
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    return result

# ==================== MAIN CODE ====================

def main():
    print("🚀 Stage 1: CVaR Optimization - Strategy Allocation")
    print("=" * 70)
    
    # Load data from CSV file
    returns_data = load_strategy_data_from_csv('data.csv')
    
    if returns_data is None:
        print("❌ Failed to load data from CSV file")
        print("💡 Make sure 'data.csv' file exists and contains correct data")
        return None
        
    print("✅ Using data from CSV file")
    
    # Remove index for compatibility with rest of the code
    returns_data = returns_data.reset_index(drop=True)
    
    # Calculate statistics
    annual_returns = returns_data.mean() * 366
    annual_volatility = returns_data.std() * np.sqrt(366)
    sharpe_ratios = annual_returns / annual_volatility
    
    print("\n📈 Strategy statistics (annual):")
    stats_df = pd.DataFrame({
        'Return': annual_returns,
        'Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratios
    }).round(3)
    print(stats_df)
    
    # CVaR optimization
    print(f"\n🎯 Running CVaR optimization (α = {1-CONFIDENCE_LEVEL:.0%})...")
    optimization_result = cvar_optimization(returns_data)
    
    if optimization_result.success:
        optimal_weights = optimization_result.x
        print("✅ Optimization successfully completed!")
    else:
        print("❌ Optimization error, using equal-weighted portfolio")
        optimal_weights = np.array([0.2] * len(STRATEGIES))
    
    # Calculate allocation
    allocations = optimal_weights * TOTAL_CAPITAL
    
    # Results
    results_df = pd.DataFrame({
        'Strategy': STRATEGIES,
        'Weight (%)': optimal_weights * 100,
        'Allocation ($)': allocations,
        'Expected Return': annual_returns.values,
        'Volatility': annual_volatility.values
    }).round(2)
    
    print("\n💰 ALLOCATION RESULTS:")
    print("=" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['Strategy']:<15}: {row['Weight (%)']:>6.1f}% | ${row['Allocation ($)']:>10,.0f}")
    
    # Portfolio metrics
    portfolio_return = np.sum(optimal_weights * annual_returns)
    portfolio_vol = np.sqrt(optimal_weights.T @ returns_data.cov().values @ optimal_weights) * np.sqrt(366)
    portfolio_sharpe = portfolio_return / portfolio_vol
    portfolio_cvar = calculate_cvar(returns_data, optimal_weights, alpha=1-CONFIDENCE_LEVEL)
    portfolio_var = calculate_var(returns_data, optimal_weights, alpha=1-CONFIDENCE_LEVEL)
    
    print(f"\n📊 PORTFOLIO METRICS:")
    print(f"Expected Return:      {portfolio_return:.1%}")
    print(f"Volatility:           {portfolio_vol:.1%}")
    print(f"Sharpe Ratio:         {portfolio_sharpe:.2f}")
    print(f"VaR (95%):           {portfolio_var:.2%}")
    print(f"CVaR (95%):          {portfolio_cvar:.2%}")
    
    # Visualization
    create_visualizations(results_df, returns_data, optimal_weights)
    
    # Save results for next stages
    stage1_results = {
        'strategy_allocations': dict(zip(STRATEGIES, allocations)),
        'strategy_weights': dict(zip(STRATEGIES, optimal_weights)),
        'total_capital': TOTAL_CAPITAL,
        'portfolio_metrics': {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': portfolio_sharpe,
            'var_95': portfolio_var,
            'cvar_95': portfolio_cvar
        }
    }
    
    # Export to CSV for further use
    results_df.to_csv('stage1_strategy_allocation.csv', index=False)
    
    return stage1_results

def create_visualizations(results_df, returns_data, weights):
    """Create visualizations"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CVaR Optimization: Strategy Allocation', fontsize=16, fontweight='bold')
    
    # 1. Pie chart of allocation
    ax1.pie(results_df['Weight (%)'], labels=results_df['Strategy'], autopct='%1.1f%%', 
            colors=COLORS, startangle=90)
    ax1.set_title('Capital Allocation Between Strategies', fontweight='bold')
    
    # 2. Bar chart of allocation in dollars
    bars = ax2.bar(results_df['Strategy'], results_df['Allocation ($)'] / 1000, color=COLORS)
    ax2.set_title('Allocation in Thousands of Dollars', fontweight='bold')
    ax2.set_ylabel('Capital (k$)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add values on bars
    for bar, value in zip(bars, results_df['Allocation ($)']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${value/1000:.0f}k', ha='center', va='bottom', fontweight='bold')
    
    # 3. Risk-return scatter plot
    ax3.scatter(results_df['Volatility'], results_df['Expected Return'], 
               s=results_df['Weight (%)'] * 10, c=COLORS, alpha=0.7)
    for i, strategy in enumerate(results_df['Strategy']):
        ax3.annotate(strategy, (results_df['Volatility'].iloc[i], results_df['Expected Return'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Volatility')
    ax3.set_ylabel('Expected Return')
    ax3.set_title('Risk-Return Profile of Strategies\n(size = weight in portfolio)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation matrix
    corr_matrix = returns_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Strategy Correlation Matrix', fontweight='bold')
    
    plt.tight_layout()
    # Save overview chart panel
    try:
        fig.savefig('stage1_overview.png', dpi=200, bbox_inches='tight')
    except Exception:
        pass
    plt.show()
    
    # Additional chart: cumulative returns
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative_returns = (1 + returns_data).cumprod()
    
    for i, strategy in enumerate(STRATEGIES):
        ax.plot(cumulative_returns.index, cumulative_returns[strategy], 
               label=strategy, color=COLORS[i], linewidth=2)
    
    # Portfolio
    portfolio_returns = returns_data @ weights
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    ax.plot(portfolio_cumulative.index, portfolio_cumulative, 
           label='Optimal Portfolio', color='black', linewidth=3, linestyle='--')
    
    ax.set_title('Cumulative Returns of Strategies', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Cumulative Returns')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save cumulative returns chart
    try:
        fig.savefig('stage1_cumulative_returns.png', dpi=200, bbox_inches='tight')
    except Exception:
        pass
    plt.show()

if __name__ == "__main__":
    results = main()
    print(f"\n✅ Stage 1 completed. Results saved to 'stage1_strategy_allocation.csv'")