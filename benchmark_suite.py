
import numpy as np
import torch
import torch.nn as nn
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector
import mitiq
from mitiq import zne
import utils
from data_gen_advanced import QEMGraphBuilder, DATASET_DIR
from models.qem_former import QEMFormer
import os
import json
from datetime import datetime

def calculate_expectation_z0(counts):
    """Calculate <Z_0> from measurement counts."""
    z0 = 0
    total = 0
    for b, c in counts.items():
        val = 1 if b[-1] == '0' else -1
        z0 += val * c
        total += c
    return z0 / total if total > 0 else 0

def calculate_ideal_z0_statevector(qc):
    """
    Calculate exact <Z_0> using statevector simulation.
    Works for ALL circuit types (Clifford, QAOA, Variational).
    """
    try:
        # Remove measurements for statevector simulation
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        
        # Get statevector
        sv = Statevector.from_instruction(qc_no_meas)
        probs = sv.probabilities()
        
        # Calculate <Z_0>
        n_qubits = qc_no_meas.num_qubits
        z0 = 0
        for i, p in enumerate(probs):
            # i is the basis state index, check qubit 0 (least significant bit)
            bit_0 = (i >> 0) & 1
            sign = 1 if bit_0 == 0 else -1
            z0 += sign * p
        
        return z0
    except Exception as e:
        print(f"  Statevector failed: {e}")
        return None

def benchmark_models(model_path="weights/qem_former.pth", save_results=True):
    """
    Benchmarks the trained QEM-Former against Noisy Baseline and Mitiq ZNE.
    Uses STATEVECTOR for accurate ground truth on all circuit types.
    """
    print("=" * 70)
    print("QEM Benchmark Suite - Team 15 (FIXED)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("Using STATEVECTOR for accurate ground truth on all circuits")
    print()
    
    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    global_dim = 5
    model = QEMFormer(noise_context_dim=global_dim).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded QEM-Former weights from: {model_path}")
        model.eval()
    else:
        print(f"WARNING: Model weights not found at {model_path}. Using random weights.")
    
    # 2. Setup Simulators
    # Use statevector for ideal (works for ALL circuits)
    # Use noisy Aer for noisy execution
    noise_model = utils.build_noise_model(scale=1.5)
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    builder = QEMGraphBuilder()
    
    def executor(circuit):
        """Executor for Mitiq ZNE"""
        t_qc = transpile(circuit, sim_noisy)
        res = sim_noisy.run(t_qc, shots=2000).result().get_counts()
        return calculate_expectation_z0(res)

    # 3. Define Test Suites
    test_suites = {
        'in_distribution': {
            'name': 'In-Distribution (Random Clifford)',
            'generator': lambda: utils.create_random_clifford_circuit(5, 20),
            'n_circuits': 30
        },
        'ood_qaoa': {
            'name': 'OOD: QAOA Circuits',
            'generator': lambda: utils.create_qaoa_circuit(5, p=2),
            'n_circuits': 20
        },
        'ood_variational': {
            'name': 'OOD: Variational Ansatz',
            'generator': lambda: utils.create_variational_circuit(5, 5),
            'n_circuits': 20
        }
    }
    
    all_results = {}
    
    for suite_key, suite_config in test_suites.items():
        print(f"\n{'='*60}")
        print(f"Test Suite: {suite_config['name']}")
        print(f"{'='*60}")
        
        results = {
            'noisy_errors': [],
            'zne_errors': [],
            'qem_errors': [],
            'improvement_ratios': [],
            'zne_vs_noisy_ratios': [],
            'ideal_values': [],
            'noisy_values': [],
            'qem_values': []
        }
        
        for i in range(suite_config['n_circuits']):
            qc, _ = suite_config['generator']()
            qc.measure_all()
            
            # A. IDEAL using STATEVECTOR (accurate for all circuit types!)
            val_ideal = calculate_ideal_z0_statevector(qc)
            if val_ideal is None:
                print(f"  [{i+1:2d}] Skipped (statevector failed)")
                continue
            
            # B. Raw Noisy
            val_noisy = executor(qc)
            
            # C. ZNE Mitigated
            try:
                fac = zne.inference.LinearFactory(scale_factors=[1.0, 2.0, 3.0])
                val_zne = zne.execute_with_zne(qc, executor, factory=fac, 
                                                scale_noise=zne.scaling.fold_gates_at_random)
            except Exception as e:
                val_zne = val_noisy  # Fallback
            
            # D. QEM-Former
            zz_noisy = 0.0
            global_attr = [val_noisy, zz_noisy, 5.0, 20.0, 1.5]
            
            graph = builder.circuit_to_graph(qc, global_features=global_attr).to(device)
            
            with torch.no_grad():
                batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(device)
                pred = model(graph.x, graph.edge_index, batch, graph.global_attr.unsqueeze(0))
                val_qem = pred.item()
            
            # Metrics
            err_noisy = abs(val_noisy - val_ideal)
            err_zne = abs(val_zne - val_ideal)
            err_qem = abs(val_qem - val_ideal)
            
            results['noisy_errors'].append(err_noisy)
            results['zne_errors'].append(err_zne)
            results['qem_errors'].append(err_qem)
            results['ideal_values'].append(val_ideal)
            results['noisy_values'].append(val_noisy)
            results['qem_values'].append(val_qem)
            
            ir_qem = err_noisy / (err_qem + 1e-9)
            ir_zne = err_noisy / (err_zne + 1e-9)
            results['improvement_ratios'].append(ir_qem)
            results['zne_vs_noisy_ratios'].append(ir_zne)
            
            # Status indicator
            status = "✓" if err_qem < err_noisy else "✗"
            print(f"  [{i+1:2d}] {status} Ideal={val_ideal:+.3f} Noisy={val_noisy:+.3f} ZNE={val_zne:+.3f} QEM={val_qem:+.3f} | IR={ir_qem:.2f}x")
        
        # Summary for this suite
        if len(results['qem_errors']) > 0:
            mean_noisy = np.mean(results['noisy_errors'])
            mean_zne = np.mean(results['zne_errors'])
            mean_qem = np.mean(results['qem_errors'])
            mean_ir = np.mean(results['improvement_ratios'])
            
            # Win rate: how often QEM beats noisy
            win_rate = sum(1 for n, q in zip(results['noisy_errors'], results['qem_errors']) if q < n) / len(results['qem_errors'])
            
            print(f"\n--- {suite_config['name']} Summary ---")
            print(f"  Mean Noisy Error:   {mean_noisy:.4f}")
            print(f"  Mean ZNE Error:     {mean_zne:.4f}")
            print(f"  Mean QEM Error:     {mean_qem:.4f}")
            print(f"  Mean IR (QEM):      {mean_ir:.2f}x")
            print(f"  Win Rate (QEM):     {win_rate*100:.1f}%")
            
            if mean_qem < mean_noisy:
                improvement_pct = (1 - mean_qem/mean_noisy) * 100
                print(f"  ✅ QEM reduces error by {improvement_pct:.1f}%")
            else:
                print(f"  ⚠️ QEM needs improvement on this circuit type")
            
            all_results[suite_key] = {
                'name': suite_config['name'],
                'n_circuits': len(results['qem_errors']),
                'mean_noisy_error': float(mean_noisy),
                'mean_zne_error': float(mean_zne),
                'mean_qem_error': float(mean_qem),
                'mean_ir_qem': float(mean_ir),
                'win_rate': float(win_rate),
                'std_qem_error': float(np.std(results['qem_errors'])),
            }
    
    # Global Summary
    print("\n" + "=" * 70)
    print("GLOBAL BENCHMARK SUMMARY (ACCURATE)")
    print("=" * 70)
    
    for suite_key, data in all_results.items():
        print(f"\n{data['name']}:")
        print(f"  QEM Error: {data['mean_qem_error']:.4f} ± {data['std_qem_error']:.4f}")
        print(f"  Improvement Ratio: {data['mean_ir_qem']:.2f}x")
        print(f"  Win Rate: {data['win_rate']*100:.1f}%")
    
    # Generalization gap with accurate metrics
    if 'in_distribution' in all_results and 'ood_qaoa' in all_results:
        gap = all_results['ood_qaoa']['mean_qem_error'] - all_results['in_distribution']['mean_qem_error']
        print(f"\nGeneralization Gap (QAOA - Clifford): {gap:+.4f}")
    
    if 'in_distribution' in all_results and 'ood_variational' in all_results:
        gap = all_results['ood_variational']['mean_qem_error'] - all_results['in_distribution']['mean_qem_error']
        print(f"Generalization Gap (Variational - Clifford): {gap:+.4f}")
    
    # Save results
    if save_results:
        results_path = "assets/benchmark_results.json"
        all_results['timestamp'] = datetime.now().isoformat()
        all_results['note'] = "Ground truth computed via statevector simulation (accurate)"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QEM Benchmark Suite (Fixed)")
    parser.add_argument("--model", type=str, default="weights/qem_former.pth", help="Model weights path")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON")
    
    args = parser.parse_args()
    
    benchmark_models(model_path=args.model, save_results=not args.no_save)
