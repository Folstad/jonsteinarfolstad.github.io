"""
Generate a cost comparison chart for Cloud LLMs vs Self-Hosted Models
"""
import matplotlib.pyplot as plt
import numpy as np

# Set up the plot style
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Generate data points
requests = np.linspace(0, 100000, 1000)  # Number of requests (in thousands)

# Cloud API costs (linear scaling)
# Assume $0.002 per request (typical API pricing)
cloud_cost = requests * 0.002

# Self-hosted costs (high initial investment, then flat marginal cost)
# Initial investment: $10,000 (hardware, setup)
# Marginal cost per request: $0.0002 (much lower than cloud)
initial_investment = 10000
self_hosted_cost = initial_investment + (requests * 0.0002)

# Plot the lines
ax.plot(requests/1000, cloud_cost, linewidth=3, label='Cloud LLM (API)', 
        color='#4285F4', linestyle='-')
ax.plot(requests/1000, self_hosted_cost, linewidth=3, label='Self-Hosted Fine-Tuned Model', 
        color='#34A853', linestyle='-')

# Find and mark the break-even point
break_even_idx = np.argmin(np.abs(cloud_cost - self_hosted_cost))
break_even_requests = requests[break_even_idx]
break_even_cost = cloud_cost[break_even_idx]

# Add break-even point marker
ax.plot(break_even_requests/1000, break_even_cost, 'ro', markersize=10, 
        label=f'Break-even Point (~{break_even_requests/1000:.0f}K requests)', zorder=5)
ax.axvline(x=break_even_requests/1000, color='red', linestyle='--', alpha=0.3)
ax.axhline(y=break_even_cost, color='red', linestyle='--', alpha=0.3)

# Add annotations
ax.annotate('Linear Growth\n(Pay per token)', 
            xy=(80, cloud_cost[800]), 
            fontsize=11, 
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#4285F4', alpha=0.2))

ax.annotate('Flat Cost Curve\n(After initial investment)', 
            xy=(80, self_hosted_cost[800]), 
            fontsize=11, 
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#34A853', alpha=0.2))

ax.annotate(f'Break-even:\n{break_even_requests/1000:.1f}K requests\n${break_even_cost:,.0f}', 
            xy=(break_even_requests/1000, break_even_cost), 
            xytext=(break_even_requests/1000 + 15, break_even_cost - 3000),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red'))

# Labels and title
ax.set_xlabel('Request Volume (thousands)', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Cost ($)', fontsize=13, fontweight='bold')
ax.set_title('Cloud LLMs vs. Fine-Tuned Models: Cost Comparison Over Scale', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid
ax.grid(True, alpha=0.3)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Format y-axis as currency
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add explanatory text box
textstr = 'Key Insights:\n• Cloud APIs: Low entry cost, scales linearly with usage\n• Self-Hosted: High upfront investment, low marginal cost\n• Break-even point determines optimal choice'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Tight layout
plt.tight_layout()

# Save the figure
output_path = 'c:/Dev/jonsteinarfolstad.github.io/images/llm-comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Chart saved to: {output_path}")

# Also save as high-res version
output_path_hd = 'c:/Dev/jonsteinarfolstad.github.io/images/llm-comparison-hd.png'
plt.savefig(output_path_hd, dpi=600, bbox_inches='tight', facecolor='white')
print(f"High-res chart saved to: {output_path_hd}")

plt.show()
print("\nChart generation complete!")
