"""
Generate Post-Hackathon Improvement Figures
Compares original 7K dataset results with new 25K dataset results
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12

# Create output directory
os.makedirs('assets/post_hackathon', exist_ok=True)

# Data: Before (7K) vs After (25K)
circuit_types = ['Clifford', 'QAOA', 'Variational']

# Original results (from hackathon)
win_rates_before = [66.7, 15.0, 80.0]
error_reduction_before = [31.2, -115.0, 31.9]

# New results (25K samples, 35% QAOA)
win_rates_after = [86.7, 95.0, 85.0]
error_reduction_after = [72.7, 95.9, 63.8]

# Improvement ratios
ir_before = [1.45, 0.46, 1.47]
ir_after = [5.76, 68.21, 10.61]

# Figure 1: Win Rate Comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(circuit_types))
width = 0.35

bars1 = ax.bar(x - width/2, win_rates_before, width, label='Before (7K samples)', color='#E74C3C', alpha=0.8)
bars2 = ax.bar(x + width/2, win_rates_after, width, label='After (25K samples)', color='#27AE60', alpha=0.8)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
               xy=(bar.get_x() + bar.get_width() / 2, height),
               xytext=(0, 3), textcoords="offset points",
               ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
               xy=(bar.get_x() + bar.get_width() / 2, height),
               xytext=(0, 3), textcoords="offset points",
               ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
ax.set_xlabel('Circuit Type')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate Improvement: 7K vs 25K Training Samples')
ax.set_xticks(x)
ax.set_xticklabels(circuit_types)
ax.legend()
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('assets/post_hackathon/win_rate_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: assets/post_hackathon/win_rate_comparison.png")

# Figure 2: QAOA Fix Highlight
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# QAOA Before/After
ax1 = axes[0]
categories = ['Win Rate', 'Error Reduction']
before_vals = [15.0, -115.0]
after_vals = [95.0, 95.9]
x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, before_vals, width, label='Before', color='#E74C3C', alpha=0.8)
bars2 = ax1.bar(x + width/2, after_vals, width, label='After', color='#27AE60', alpha=0.8)

for bar, val in zip(bars1, before_vals):
    height = bar.get_height()
    ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, max(height, 0) + 5),
                ha='center', fontsize=11)
                
for bar, val in zip(bars2, after_vals):
    height = bar.get_height()
    ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height + 5),
                ha='center', fontsize=11, fontweight='bold')

ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.set_ylabel('Percentage (%)')
ax1.set_title('QAOA Circuit Performance Fix')
ax1.legend()
ax1.set_ylim(-130, 120)

# Improvement Ratio comparison
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, [ir_before[1], ir_after[1]/10], width, label='', color=['#E74C3C', '#27AE60'], alpha=0.8)

models = ['Before\n(7K)', 'After\n(25K)']
irs = [ir_before[1], ir_after[1]]
colors = ['#E74C3C', '#27AE60']

bars = ax2.bar(models, irs, color=colors, alpha=0.8)
for bar, ir in zip(bars, irs):
    ax2.annotate(f'{ir:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 1),
                ha='center', fontsize=12, fontweight='bold')

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Improvement (1x)')
ax2.set_ylabel('Improvement Ratio')
ax2.set_title('QAOA Improvement Ratio')
ax2.legend()

plt.tight_layout()
plt.savefig('assets/post_hackathon/qaoa_fix.png', bbox_inches='tight')
plt.close()
print("Saved: assets/post_hackathon/qaoa_fix.png")

# Figure 3: Overall Summary Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(circuit_types))
width = 0.35

bars1 = ax.bar(x - width/2, ir_before, width, label='Before (7K samples)', color='#E74C3C', alpha=0.8)
bars2 = ax.bar(x + width/2, ir_after, width, label='After (25K samples)', color='#27AE60', alpha=0.8)

# Add value labels
for bar, ir in zip(bars1, ir_before):
    ax.annotate(f'{ir:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
               ha='center', fontsize=10)

for bar, ir in zip(bars2, ir_after):
    ax.annotate(f'{ir:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
               ha='center', fontsize=10, fontweight='bold')

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Improvement (1x)')
ax.set_xlabel('Circuit Type')
ax.set_ylabel('Improvement Ratio')
ax.set_title('Improvement Ratio: Before vs After Dataset Enhancement')
ax.set_xticks(x)
ax.set_xticklabels(circuit_types)
ax.legend()
ax.set_ylim(0, max(ir_after) * 1.2)

plt.tight_layout()
plt.savefig('assets/post_hackathon/improvement_ratio_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: assets/post_hackathon/improvement_ratio_comparison.png")

print("\n=== All post-hackathon figures generated ===")
