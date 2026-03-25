"""
Enhanced Benchmark Suite for CDRFormer - IEEE QCE 2026

Implements proper baselines with statistical rigor:
- ZNE (Linear + Richardson factories)
- CDR (Linear regression baseline)
- Multi-seed statistical analysis
- Confidence intervals
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector
import mitiq
from mitiq import zne
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from sklearn.linear_model import LinearRegression
import utils
from data_gen_advanced import QEMGraphBuilder, DATASET_DIR
from models.cdr_former import CDRFormer
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
RANDOM_SEEDS = [42, 123, 456, 789, 2024]  # For reproducibility
NOISE_SCALE = 1.5
DEFAULT_SHOTS = 2000

# ============================================================
# Core Functions
# ============================================================

def calculate_expectation_z0(counts: dict) -> float:
    """Calculate <Z_0> from measurement counts."""
    z0 = 0
    total = 0
    for b, c in counts.items():
        val = 1 if b[-1] == '0' else -1
        z0 += val * c
        total += c
    return z0 / total if total > 0 else 0

def calculate_ideal_z0_statevector(qc) -> Optional[float]:
    """
    Calculate exact <Z_0> using statevector simulation.
    Works for ALL circuit types (Clifford, QAOA, Variational).
    """
    try:
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        probs = sv.probabilities()
        
        z0 = 0
        for i, p in enumerate(probs):
            bit_0 = (i >> 0) & 1
            sign = 1 if bit_0 == 0 else -1
            z0 += sign * p
        
        return z0
    except Exception as e:
        return None

# ============================================================
# CDR Baseline (Linear Regression - No ML)
# ============================================================

def train_cdr_baseline(n_training_circuits: int = 50, 
                        noise_model=None,
                        n_qubits: int = 5,
                        depth: int = 20) -> LinearRegression:
    """
    Train a simple CDR baseline using linear regression.
    This is what CDR [Czarnik21] does without graph neural networks.
    
    Uses Clifford circuits where ideal values can be computed exactly.
    """
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    X_train = []  # Noisy values
    y_train = []  # Ideal values
    
    for _ in range(n_training_circuits):
        qc, _ = utils.create_random_clifford_circuit(n_qubits, depth)
        qc.measure_all()
        
        # Get ideal value (exact for Clifford)
        val_ideal = calculate_ideal_z0_statevector(qc)
        if val_ideal is None:
            continue
        
        # Get noisy value
        t_qc = transpile(qc, basis_gates=sim_noisy.operation_names)
        res = sim_noisy.run(t_qc, shots=DEFAULT_SHOTS).result().get_counts()
        val_noisy = calculate_expectation_z0(res)
        
        X_train.append([val_noisy])
        y_train.append(val_ideal)
    
    # Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def apply_cdr_baseline(model: LinearRegression, noisy_value: float) -> float:
    """Apply trained CDR baseline to correct a noisy value."""
    return model.predict([[noisy_value]])[0]


# ============================================================
# Enhanced Benchmark with All Baselines
# ============================================================

def benchmark_single_seed(seed: int, 
                          model_path: str,
                          test_suites: dict,
                          verbose: bool = True) -> dict:
    """
    Run full benchmark with a single random seed.
    Returns detailed results for aggregation.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running benchmark with seed: {seed}")
        print(f"{'='*60}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_model = utils.build_noise_model(scale=NOISE_SCALE)
    sim_noisy = AerSimulator(noise_model=noise_model)
    builder = QEMGraphBuilder()
    
    # Load CDRFormer
    global_dim = 5
    cdr_former = CDRFormer(noise_context_dim=global_dim).to(device)
    if os.path.exists(model_path):
        cdr_former.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        cdr_former.eval()
    
    # Train CDR baseline for this seed
    cdr_baseline = train_cdr_baseline(
        n_training_circuits=50,
        noise_model=noise_model,
        n_qubits=5,
        depth=20
    )
    
    # Executor for Mitiq
    def executor(circuit):
        t_qc = transpile(circuit, basis_gates=sim_noisy.operation_names)
        res = sim_noisy.run(t_qc, shots=DEFAULT_SHOTS).result().get_counts()
        return calculate_expectation_z0(res)
    
    all_results = {}
    
    for suite_key, suite_config in test_suites.items():
        if verbose:
            print(f"\n--- {suite_config['name']} ---")
        
        results = {
            'noisy_errors': [],
            'zne_linear_errors': [],
            'zne_richardson_errors': [],
            'cdr_baseline_errors': [],
            'cdr_former_errors': [],
            'ideal_values': [],
            'noisy_values': [],
        }
        
        for i in range(suite_config['n_circuits']):
            qc, _ = suite_config['generator']()
            qc.measure_all()
            
            # IDEAL
            val_ideal = calculate_ideal_z0_statevector(qc)
            if val_ideal is None:
                continue
            
            # NOISY
            val_noisy = executor(qc)
            
            # ZNE - Linear Factory
            try:
                fac_linear = LinearFactory(scale_factors=[1.0, 2.0, 3.0])
                val_zne_linear = zne.execute_with_zne(
                    qc, executor, factory=fac_linear,
                    scale_noise=zne.scaling.fold_gates_at_random
                )
            except:
                val_zne_linear = val_noisy
            
            # ZNE - Richardson Factory (higher order)
            try:
                fac_rich = RichardsonFactory(scale_factors=[1.0, 2.0, 3.0])
                val_zne_richardson = zne.execute_with_zne(
                    qc, executor, factory=fac_rich,
                    scale_noise=zne.scaling.fold_gates_at_random
                )
            except:
                val_zne_richardson = val_noisy
            
            # CDR Baseline (Linear Regression)
            val_cdr_baseline = apply_cdr_baseline(cdr_baseline, val_noisy)
            
            # CDRFormer (Our Model)
            zz_noisy = 0.0
            global_attr = [val_noisy, zz_noisy, 5.0, 20.0, NOISE_SCALE]
            graph = builder.circuit_to_graph(qc, global_features=global_attr).to(device)
            
            with torch.no_grad():
                batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(device)
                pred = cdr_former(graph.x, graph.edge_index, batch, graph.global_attr.unsqueeze(0))
                val_cdr_former = pred.item()
            
            # Store results
            results['ideal_values'].append(val_ideal)
            results['noisy_values'].append(val_noisy)
            results['noisy_errors'].append(abs(val_noisy - val_ideal))
            results['zne_linear_errors'].append(abs(val_zne_linear - val_ideal))
            results['zne_richardson_errors'].append(abs(val_zne_richardson - val_ideal))
            results['cdr_baseline_errors'].append(abs(val_cdr_baseline - val_ideal))
            results['cdr_former_errors'].append(abs(val_cdr_former - val_ideal))
            
            if verbose and i < 5:  # Show first 5 only
                print(f"  [{i+1}] Ideal={val_ideal:+.3f} Noisy={val_noisy:+.3f} "
                      f"ZNE={val_zne_linear:+.3f} CDR-LR={val_cdr_baseline:+.3f} "
                      f"CDRFormer={val_cdr_former:+.3f}")
        
        # Compute summary statistics
        if len(results['noisy_errors']) > 0:
            all_results[suite_key] = {
                'name': suite_config['name'],
                'n_circuits': len(results['noisy_errors']),
                'noisy': {
                    'mean_error': float(np.mean(results['noisy_errors'])),
                    'std_error': float(np.std(results['noisy_errors'])),
                },
                'zne_linear': {
                    'mean_error': float(np.mean(results['zne_linear_errors'])),
                    'std_error': float(np.std(results['zne_linear_errors'])),
                    'win_rate': float(np.mean([z < n for z, n in zip(results['zne_linear_errors'], results['noisy_errors'])])),
                },
                'zne_richardson': {
                    'mean_error': float(np.mean(results['zne_richardson_errors'])),
                    'std_error': float(np.std(results['zne_richardson_errors'])),
                    'win_rate': float(np.mean([z < n for z, n in zip(results['zne_richardson_errors'], results['noisy_errors'])])),
                },
                'cdr_baseline': {
                    'mean_error': float(np.mean(results['cdr_baseline_errors'])),
                    'std_error': float(np.std(results['cdr_baseline_errors'])),
                    'win_rate': float(np.mean([c < n for c, n in zip(results['cdr_baseline_errors'], results['noisy_errors'])])),
                },
                'cdr_former': {
                    'mean_error': float(np.mean(results['cdr_former_errors'])),
                    'std_error': float(np.std(results['cdr_former_errors'])),
                    'win_rate': float(np.mean([c < n for c, n in zip(results['cdr_former_errors'], results['noisy_errors'])])),
                }
            }
    
    return all_results


def aggregate_multi_seed_results(all_seed_results: List[dict]) -> dict:
    """
    Aggregate results across multiple seeds.
    Computes mean ± std for all metrics.
    """
    aggregated = {}
    
    # Get all suite keys
    suite_keys = list(all_seed_results[0].keys())
    
    for suite_key in suite_keys:
        suite_data = [r[suite_key] for r in all_seed_results if suite_key in r]
        
        if len(suite_data) == 0:
            continue
        
        methods = ['noisy', 'zne_linear', 'zne_richardson', 'cdr_baseline', 'cdr_former']
        
        aggregated[suite_key] = {
            'name': suite_data[0]['name'],
            'n_seeds': len(suite_data),
            'methods': {}
        }
        
        for method in methods:
            mean_errors = [s[method]['mean_error'] for s in suite_data]
            
            aggregated[suite_key]['methods'][method] = {
                'mean_error': float(np.mean(mean_errors)),
                'std_across_seeds': float(np.std(mean_errors)),
                'ci_95': float(1.96 * np.std(mean_errors) / np.sqrt(len(mean_errors))),
            }
            
            if method != 'noisy' and 'win_rate' in suite_data[0][method]:
                win_rates = [s[method]['win_rate'] for s in suite_data]
                aggregated[suite_key]['methods'][method]['win_rate'] = float(np.mean(win_rates))
                aggregated[suite_key]['methods'][method]['win_rate_std'] = float(np.std(win_rates))
    
    return aggregated


def run_full_benchmark(model_path: str = "weights/cdr_former.pth",
                       n_seeds: int = 5,
                       save_results: bool = True) -> dict:
    """
    Run comprehensive benchmark with multiple random seeds.
    Produces publication-ready statistics with confidence intervals.
    """
    print("=" * 70)
    print("CDRFormer Enhanced Benchmark Suite")
    print("IEEE QCE 2026 - Publication-Ready Statistics")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Random Seeds: {RANDOM_SEEDS[:n_seeds]}")
    print(f"Model: {model_path}")
    print()
    
    # Define test suites
    test_suites = {
        'clifford': {
            'name': 'Random Clifford (In-Distribution)',
            'generator': lambda: utils.create_random_clifford_circuit(5, 20),
            'n_circuits': 30
        },
        'qaoa': {
            'name': 'QAOA MaxCut',
            'generator': lambda: utils.create_qaoa_circuit(5, p=2),
            'n_circuits': 20
        },
        'variational': {
            'name': 'Variational Ansatz (HEA)',
            'generator': lambda: utils.create_variational_circuit(5, 5),
            'n_circuits': 20
        }
    }
    
    # Run for each seed
    all_seed_results = []
    for i, seed in enumerate(RANDOM_SEEDS[:n_seeds]):
        print(f"\n[Seed {i+1}/{n_seeds}] Running with seed {seed}...")
        results = benchmark_single_seed(
            seed=seed,
            model_path=model_path,
            test_suites=test_suites,
            verbose=(i == 0)  # Only verbose for first seed
        )
        all_seed_results.append(results)
    
    # Aggregate
    aggregated = aggregate_multi_seed_results(all_seed_results)
    
    # Print Summary Table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (Mean ± Std across seeds)")
    print("=" * 70)
    
    for suite_key, data in aggregated.items():
        print(f"\n{data['name']}:")
        print("-" * 50)
        print(f"{'Method':<20} {'MAE':>12} {'Win Rate':>12}")
        print("-" * 50)
        
        noisy_mae = data['methods']['noisy']['mean_error']
        print(f"{'Noisy (Baseline)':<20} {noisy_mae:.4f} ± {data['methods']['noisy']['std_across_seeds']:.4f}")
        
        for method in ['zne_linear', 'zne_richardson', 'cdr_baseline', 'cdr_former']:
            m = data['methods'][method]
            name_map = {
                'zne_linear': 'ZNE (Linear)',
                'zne_richardson': 'ZNE (Richardson)',
                'cdr_baseline': 'CDR (Linear Reg)',
                'cdr_former': 'CDRFormer (Ours)'
            }
            win_str = f"{m['win_rate']*100:.1f}%" if 'win_rate' in m else "N/A"
            print(f"{name_map[method]:<20} {m['mean_error']:.4f} ± {m['std_across_seeds']:.4f} {win_str:>12}")
        
        # Improvement over best baseline
        baselines = ['zne_linear', 'zne_richardson', 'cdr_baseline']
        best_baseline = min(baselines, key=lambda x: data['methods'][x]['mean_error'])
        best_baseline_mae = data['methods'][best_baseline]['mean_error']
        our_mae = data['methods']['cdr_former']['mean_error']
        
        if our_mae < best_baseline_mae:
            improvement = (1 - our_mae / best_baseline_mae) * 100
            print(f"\n✅ CDRFormer improves over best baseline ({best_baseline}) by {improvement:.1f}%")
        else:
            print(f"\n⚠️ Best baseline: {best_baseline}")
    
    # Save
    if save_results:
        results_path = "assets/benchmark_results_publication.json"
        aggregated['timestamp'] = datetime.now().isoformat()
        aggregated['n_seeds'] = n_seeds
        aggregated['seeds_used'] = RANDOM_SEEDS[:n_seeds]
        with open(results_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"\n📊 Results saved to: {results_path}")
    
    return aggregated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CDRFormer Enhanced Benchmark Suite")
    parser.add_argument("--model", type=str, default="weights/cdr_former.pth", help="Model weights path")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (1-5)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON")
    
    args = parser.parse_args()
    
    run_full_benchmark(
        model_path=args.model,
        n_seeds=min(args.seeds, len(RANDOM_SEEDS)),
        save_results=not args.no_save
    )
