"""
CDRFormer Publication Figures

Generates publication-quality figures from benchmark results:
1. Data Composition (QAOA% vs Win Rate)
2. Scalability (Qubits vs Win Rate)
3. Noise Profile Comparison
4. Baseline Comparison Bar Chart
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Style settings for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

COLORS = {
    'primary': '#2E86AB',    # Blue
    'secondary': '#A23B72',  # Magenta
    'success': '#28A745',    # Green
    'warning': '#FFC107',    # Yellow
    'danger': '#DC3545',     # Red
}


def fig_data_composition():
    """Figure 1: QAOA Training Ratio vs Win Rate"""
    with open('assets/data_composition_ablation.json', 'r') as f:
        data = json.load(f)
    
    # Extract data
    configs = ['low_qaoa', 'medium_qaoa', 'high_qaoa', 'very_high_qaoa']
    qaoa_pcts = [8, 20, 35, 50]
    win_rates = [data[c]['qaoa_win_rate'] * 100 for c in configs]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    bars = ax.bar(qaoa_pcts, win_rates, width=8, color=COLORS['primary'], edgecolor='black', linewidth=0.5)
    
    # Highlight the threshold
    ax.axhline(y=100, color=COLORS['success'], linestyle='--', alpha=0.7, label='Perfect (100%)')
    ax.axhline(y=93.3, color=COLORS['warning'], linestyle=':', alpha=0.7)
    
    # Add value labels
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('QAOA Training Data Ratio (%)')
    ax.set_ylabel('QAOA Win Rate (%)')
    ax.set_title('Impact of Training Data Composition on QAOA Performance')
    ax.set_ylim(0, 110)
    ax.set_xticks(qaoa_pcts)
    
    # Add annotation
    ax.annotate('≥20% QAOA → 100%\nwin rate', xy=(20, 100), xytext=(35, 85),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('assets/fig_data_composition.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/fig_data_composition.pdf', bbox_inches='tight')
    print("Saved: fig_data_composition.png/pdf")
    plt.close()


def fig_scalability():
    """Figure 2: Qubit Scaling Performance"""
    with open('assets/scalability_benchmark.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    qubits = [int(q) for q in sorted(results.keys(), key=int)]
    win_rates = [results[str(q)]['win_rate'] * 100 for q in qubits]
    inference_times = [results[str(q)]['avg_inference_ms'] for q in qubits]
    
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    # Win rate bars
    bars = ax1.bar(qubits, win_rates, width=2, color=COLORS['primary'], 
                   edgecolor='black', linewidth=0.5, label='Win Rate')
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Win Rate (%)', color=COLORS['primary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.set_ylim(0, 110)
    
    # Inference time line
    ax2 = ax1.twinx()
    ax2.plot(qubits, inference_times, 'o-', color=COLORS['secondary'], 
             linewidth=2, markersize=6, label='Inference Time')
    ax2.set_ylabel('Inference Time (ms)', color=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax2.set_ylim(0, 3)
    
    # Training range annotation
    ax1.axvspan(4, 6, alpha=0.2, color='green', label='Training Range')
    
    ax1.set_title('CDRFormer Scalability: 5-18 Qubits')
    ax1.set_xticks(qubits)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    
    plt.tight_layout()
    plt.savefig('assets/fig_scalability.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/fig_scalability.pdf', bbox_inches='tight')
    print("Saved: fig_scalability.png/pdf")
    plt.close()


def fig_noise_profiles():
    """Figure 3: Performance Across Noise Profiles"""
    with open('assets/noise_profile_benchmark.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    profiles = list(results.keys())
    win_rates = [results[p]['win_rate'] * 100 for p in profiles]
    
    # Shorten names
    short_names = ['Incoherent', 'Coherent', 'Combined', 'FakeHanoi', 'FakeCairo'][:len(profiles)]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    colors = [COLORS['primary']] * 3 + [COLORS['secondary']] * 2
    colors = colors[:len(profiles)]
    
    bars = ax.barh(short_names, win_rates, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{rate:.1f}%', ha='left', va='center', fontsize=10)
    
    ax.set_xlabel('Win Rate (%)')
    ax.set_title('CDRFormer Performance Across Noise Profiles')
    ax.set_xlim(0, 115)
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('assets/fig_noise_profiles.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/fig_noise_profiles.pdf', bbox_inches='tight')
    print("Saved: fig_noise_profiles.png/pdf")
    plt.close()


def fig_baseline_comparison():
    """Figure 4: Comparison with Baselines"""
    methods = ['Noisy', 'ZNE\n(Linear)', 'ZNE\n(Richardson)', 'CDR\n(Linear Reg)', 'CDRFormer\n(Ours)']
    # From benchmark_results_publication.json - QAOA circuit results
    qaoa_win_rates = [0, 51.0, 21.0, 38.0, 98.0]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    colors = ['gray', COLORS['warning'], COLORS['warning'], COLORS['secondary'], COLORS['success']]
    bars = ax.bar(methods, qaoa_win_rates, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, rate in zip(bars, qaoa_win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('QAOA Win Rate (%)')
    ax.set_title('CDRFormer vs Baseline Methods on QAOA Circuits')
    ax.set_ylim(0, 115)
    
    # Highlight our method
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('assets/fig_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/fig_baseline_comparison.pdf', bbox_inches='tight')
    print("Saved: fig_baseline_comparison.png/pdf")
    plt.close()


if __name__ == "__main__":
    print("Generating CDRFormer Publication Figures...")
    print("=" * 50)
    
    fig_data_composition()
    fig_scalability()
    fig_noise_profiles()
    fig_baseline_comparison()
    
    print("=" * 50)
    print("All figures saved to assets/ directory")
