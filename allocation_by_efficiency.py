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
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class SimplePortfolioAllocator:

    
    def __init__(self, data: List[Dict], min_weight: float = 0.04, max_weight: float = 0.18):
        self.df = pd.DataFrame(data)
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Additional metric
        self.df['efficiency'] = self.df['return'] / self.df['drawdown']
        
        print(f"✅ Assets: {len(self.df)}, Constraints: {min_weight*100:.1f}%-{max_weight*100:.1f}%")
        
    def risk_based_method(self) -> Dict[str, float]:
        """Inversely proportional to risk"""
        inverse_risk = 1 / self.df['drawdown']
        weights = inverse_risk / inverse_risk.sum()
        return dict(zip(self.df['name'], weights))
    
    def return_weighted_method(self) -> Dict[str, float]:
        """Proportional to returns"""
        weights = self.df['return'] / self.df['return'].sum()
        return dict(zip(self.df['name'], weights))
    
    def efficiency_method(self) -> Dict[str, float]:
        """By efficiency (return/risk)"""
        weights = self.df['efficiency'] / self.df['efficiency'].sum()
        return dict(zip(self.df['name'], weights))
    
    def cap_weighted_method(self) -> Dict[str, float]:
        """By market capitalization"""
        weights = self.df['cap'] / self.df['cap'].sum()
        return dict(zip(self.df['name'], weights))
    
    def apply_constraints(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints"""
        result = allocation.copy()
        
        # Simple correction in several iterations
        for _ in range(20):
            excess = 0
            deficit = 0
            free_assets = []
            
            for asset, weight in result.items():
                if weight > self.max_weight:
                    excess += weight - self.max_weight
                    result[asset] = self.max_weight
                elif weight < self.min_weight:
                    deficit += self.min_weight - weight  
                    result[asset] = self.min_weight
                else:
                    free_assets.append(asset)
            
            if excess == 0 and deficit == 0:
                break
                
            # Distribute excess/deficit among free assets
            if free_assets:
                adjustment = (excess - deficit) / len(free_assets)
                for asset in free_assets:
                    result[asset] += adjustment
        
        # Normalization
        total = sum(result.values())
        return {k: v/total for k, v in result.items()}
    
    def calculate_final_allocation(self, method_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate final allocation as a combination of methods
        """
        # Get all methods
        methods = {
            'risk_based': self.risk_based_method(),
            'return_weighted': self.return_weighted_method(), 
            'efficiency': self.efficiency_method(),
            'cap_weighted': self.cap_weighted_method()
        }
        
        # Combine methods with weights
        final = {}
        for asset in self.df['name']:
            weight = 0
            for method_name, method_weight in method_weights.items():
                weight += methods[method_name][asset] * method_weight
            final[asset] = weight
        
        # Normalization
        total = sum(final.values())
        final = {k: v/total for k, v in final.items()}
        
        # Apply constraints
        constrained = self.apply_constraints(final)
        
        return methods, final, constrained
    
    def calculate_metrics(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Simple portfolio metrics"""
        weights = np.array([allocation[name] for name in self.df['name']])
        
        return {
            'return': np.sum(weights * self.df['return']),
            'risk': np.sum(weights * self.df['drawdown']),
            'efficiency': np.sum(weights * self.df['return']) / np.sum(weights * self.df['drawdown']),
            'concentration': np.sum(weights ** 2),
            'diversification': 1 / np.sum(weights ** 2),
            'max_weight': max(weights) * 100,
            'min_weight': min(weights) * 100
        }
    
    def create_results_table(self, methods: Dict, final: Dict, constrained: Dict) -> pd.DataFrame:
        """Create table with all results"""
        table = pd.DataFrame({
            'Asset': self.df['name'],
            'Return_%': self.df['return'],
            'Risk_%': self.df['drawdown'], 
            'Efficiency': self.df['efficiency'],
            'Risk_Based_%': [methods['risk_based'][name] * 100 for name in self.df['name']],
            'Return_Based_%': [methods['return_weighted'][name] * 100 for name in self.df['name']],
            'Efficiency_Based_%': [methods['efficiency'][name] * 100 for name in self.df['name']],
            'Cap_Based_%': [methods['cap_weighted'][name] * 100 for name in self.df['name']],
            'Combined_%': [final[name] * 100 for name in self.df['name']],
            'Final_%': [constrained[name] * 100 for name in self.df['name']]
        })
        
        return table

    def plot_allocation_comparison(self, methods: Dict, final: Dict, constrained: Dict):
        """Visualization of all methods + final allocation"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Portfolio Allocation Methods Comparison', fontsize=16, fontweight='bold')
        
        # Data for charts
        assets = list(self.df['name'])
        plot_data = [
            ('Inverse Risk', [methods['risk_based'][asset] * 100 for asset in assets]),
            ('Return Weighted', [methods['return_weighted'][asset] * 100 for asset in assets]),
            ('Efficiency Based', [methods['efficiency'][asset] * 100 for asset in assets]),
            ('Market Cap Weighted', [methods['cap_weighted'][asset] * 100 for asset in assets]),
            ('Combined', [final[asset] * 100 for asset in assets]),
            ('Final (with constraints)', [constrained[asset] * 100 for asset in assets])
        ]
        
        # Create charts
        for idx, (title, weights) in enumerate(plot_data):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            # Sort for visual appeal
            sorted_pairs = sorted(zip(assets, weights), key=lambda x: x[1], reverse=True)
            sorted_assets, sorted_weights = zip(*sorted_pairs)
            
            # Colors for final chart (red if constraint violation)
            if idx == 5:  # Final chart
                colors = ['red' if w < self.min_weight*100 or w > self.max_weight*100 else 'green' 
                         for w in sorted_weights]
            else:
                colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_assets)))
            
            bars = ax.bar(sorted_assets, sorted_weights, color=colors, alpha=0.7)
            
            # Constraint lines
            ax.axhline(y=self.min_weight*100, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axhline(y=self.max_weight*100, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Weight (%)')
            ax.tick_params(axis='x', rotation=45)
            
            # Label values greater than 2%
            for bar, weight in zip(bars, sorted_weights):
                if weight > 2:
                    ax.text(bar.get_x() + bar.get_width()/2., weight + 0.5,
                           f'{weight:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def generate_report(self, method_weights: Dict[str, float]) -> str:
        """Generate detailed report"""
        
        # Get all calculations
        methods, final, constrained = self.calculate_final_allocation(method_weights)
        
        # Create results table
        results_table = self.create_results_table(methods, final, constrained)
        
        # Portfolio metrics
        final_metrics = self.calculate_metrics(constrained)
        
        # Sort final allocation
        final_sorted = sorted(constrained.items(), key=lambda x: x[1], reverse=True)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           PORTFOLIO ALLOCATION REPORT                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 ASSET SOURCE DATA:
{'-'*80}
{results_table[['Asset', 'Return_%', 'Risk_%', 'Efficiency']].to_string(index=False, float_format='%.2f')}

📋 INDIVIDUAL METHOD RESULTS:
{'-'*80}
{results_table[['Asset', 'Risk_Based_%', 'Return_Based_%', 'Efficiency_Based_%', 'Cap_Based_%']].to_string(index=False, float_format='%.1f')}

⚖️  METHOD WEIGHTS IN COMBINATION:
{'-'*80}"""
        
        for method, weight in method_weights.items():
            method_names = {
                'risk_based': 'Inverse Risk',
                'return_weighted': 'Return Weighted', 
                'efficiency': 'Efficiency Based',
                'cap_weighted': 'Market Cap Weighted'
            }
            report += f"\n• {method_names.get(method, method)}: {weight*100:5.1f}%"
        
        report += f"""

🎯 FINAL ALLOCATION CALCULATION STAGES:
{'-'*80}
{results_table[['Asset', 'Combined_%', 'Final_%']].to_string(index=False, float_format='%.1f')}

🏆 FINAL ALLOCATION (sorted):
{'-'*80}"""
        
        for i, (asset, weight) in enumerate(final_sorted, 1):
            asset_data = self.df[self.df['name'] == asset].iloc[0]
            status = "✅" if self.min_weight <= weight <= self.max_weight else "⚠️"
            report += f"""
{i:2d}. {asset:6s}: {weight*100:5.1f}% {status} │ Return: {asset_data['return']:5.1f}% │ Risk: {asset_data['drawdown']:4.1f}%"""

        report += f"""

📈 FINAL PORTFOLIO METRICS:
{'-'*80}
Expected Return:           {final_metrics['return']:8.2f}%
Portfolio Risk:            {final_metrics['risk']:8.2f}%  
Efficiency:                {final_metrics['efficiency']:8.2f}
Concentration (HHI):       {final_metrics['concentration']:8.3f}
Diversification:           {final_metrics['diversification']:8.1f}
Maximum Weight:            {final_metrics['max_weight']:8.1f}%
Minimum Weight:            {final_metrics['min_weight']:8.1f}%

🔍 METHOD FORMULAS:
{'-'*80}
1. Inverse Risk:     w_i = (1/risk_i) / Σ(1/risk_j)
2. Return Weighted:  w_i = return_i / Σ(return_j)
3. Efficiency Based: w_i = efficiency_i / Σ(efficiency_j)
4. Market Cap:       w_i = cap_i / Σ(cap_j)

Combined:            w_i = Σ(method_weight × method_result_i)
With Constraints:    {self.min_weight*100:.1f}% ≤ w_i ≤ {self.max_weight*100:.1f}%

💡 EXPLANATIONS:
{'-'*80}
• Efficiency = Return / Risk (Sharpe-like ratio without risk-free rate)
• Concentration = Σ(weight²), lower is better for diversification
• Portfolio Risk = Σ(weight × asset_risk), simplified model without correlations
"""
        
        return report, results_table


def main():
    """Run analysis with clear results"""
    
    print("🚀 Starting Simple Portfolio Allocator")
    print("=" * 60)
    
    # Source data  
    crypto_data = [
        {'name': 'SOL', 'return': 22.62, 'drawdown': 3.85, 'cap': 75846923084},
        {'name': 'AVAX', 'return': 21.09, 'drawdown': 8.76, 'cap': 8018288278},
        {'name': 'ETH', 'return': 42.03, 'drawdown': 16.72, 'cap': 290734266974},
        {'name': 'INJ', 'return': 31.00, 'drawdown': 15.47, 'cap': 1143001496},
        {'name': 'RUNE', 'return': 32.78, 'drawdown': 24.14, 'cap': 474088429},
        {'name': 'MKR', 'return': 50.30, 'drawdown': 25.13, 'cap': 942477924},
        {'name': 'BTC', 'return': 24.24, 'drawdown': 1.27, 'cap': 2090440108182},
        {'name': 'DOGE', 'return': 43.26, 'drawdown': 4.85, 'cap': 26556169176},
        {'name': 'LINK', 'return': 39.26, 'drawdown': 3.88, 'cap': 13138138234},
        {'name': 'ADA', 'return': 45.43, 'drawdown': 7.18, 'cap': 28032348215},
    ]
    
    # Parameters
    allocator = SimplePortfolioAllocator(
        data=crypto_data,
        min_weight=0.04,  # 4%
        max_weight=0.18   # 18%
    )
    
    # Method weights (configurable)
    method_weights = {
        'risk_based': 0.25,       # 25% - conservative approach
        'return_weighted': 0.25,  # 25% - aggressive approach  
        'efficiency': 0.35,       # 35% - balanced approach
        'cap_weighted': 0.15      # 15% - market approach
    }
    
    print(f"📊 Method weights: {method_weights}")
    
    # Calculation and report
    methods, final, constrained = allocator.calculate_final_allocation(method_weights)
    report, results_table = allocator.generate_report(method_weights)
    
    # Output report
    print(report)
    
    # Visualization
    print("\n📊 Creating charts...")
    allocator.plot_allocation_comparison(methods, final, constrained)
    
    # Save results
    results_table.to_csv('portfolio_results.csv', index=False)
    print(f"\n💾 Results saved to 'portfolio_results.csv'")
    
    print(f"🎉 Analysis completed!")
    
    return allocator, constrained, results_table


if __name__ == "__main__":
    allocator, final_allocation, results = main()