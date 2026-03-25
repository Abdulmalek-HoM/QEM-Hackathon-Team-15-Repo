"""
CDRFormer Scalability Benchmark

Tests CDRFormer on larger circuits (10, 15, 20 qubits) to verify
it generalizes beyond the 5-6 qubit training range.

Key questions:
1. Does CDRFormer scale to larger circuits?
2. What's the performance vs noisy baseline at each scale?
3. What's the inference time at each scale?
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import time
from datetime import datetime
import json

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from data_gen_advanced import QEMGraphBuilder
from models.cdr_former import CDRFormer
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SHOTS = 2000


def benchmark_scale(model, n_qubits, n_circuits=20, seed=42):
    """
    Benchmark CDRFormer at a specific qubit count.
    Returns win rate, MAE, and inference time.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    noise_model = utils.build_noise_model(scale=1.5)
    sim_noisy = AerSimulator(noise_model=noise_model)
    builder = QEMGraphBuilder()
    model.eval()
    
    wins = 0
    errors_noisy = []
    errors_pred = []
    inference_times = []
    
    for i in range(n_circuits):
        # Generate QAOA circuit at this scale
        try:
            qc, _ = utils.create_qaoa_circuit(n_qubits, p=2)
            qc.measure_all()
        except Exception as e:
            print(f"Error creating circuit: {e}")
            continue
        
        # Ideal value via statevector (may be slow for large circuits)
        try:
            qc_no_meas = qc.remove_final_measurements(inplace=False)
            sv = Statevector.from_instruction(qc_no_meas)
            probs = sv.probabilities()
            val_ideal = sum((1 if (idx >> 0) & 1 == 0 else -1) * p for idx, p in enumerate(probs))
        except Exception as e:
            # Statevector too large, skip
            print(f"Statevector failed for {n_qubits} qubits: {e}")
            continue
        
        # Noisy execution
        try:
            t_qc = transpile(qc, basis_gates=sim_noisy.operation_names)
            counts = sim_noisy.run(t_qc, shots=SHOTS).result().get_counts()
            z0 = 0
            total = 0
            for b, c in counts.items():
                val = 1 if b[-1] == '0' else -1
                z0 += val * c
                total += c
            val_noisy = z0 / total if total > 0 else 0
        except Exception as e:
            continue
        
        # CDRFormer prediction with timing
        depth = 6 * n_qubits  # Approximate depth for QAOA p=2
        global_attr = [val_noisy, 0.0, float(n_qubits), float(depth), 1.5]
        graph = builder.circuit_to_graph(qc, global_features=global_attr).to(DEVICE)
        
        start_time = time.time()
        with torch.no_grad():
            batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(DEVICE)
            pred = model(graph.x, graph.edge_index, batch, graph.global_attr.unsqueeze(0))
            val_pred = pred.item()
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        err_noisy = abs(val_noisy - val_ideal)
        err_pred = abs(val_pred - val_ideal)
        
        errors_noisy.append(err_noisy)
        errors_pred.append(err_pred)
        
        if err_pred < err_noisy:
            wins += 1
    
    if not errors_noisy:
        return None
    
    return {
        'n_qubits': n_qubits,
        'n_circuits': len(errors_noisy),
        'win_rate': wins / len(errors_noisy),
        'mae_noisy': float(np.mean(errors_noisy)),
        'mae_cdrformer': float(np.mean(errors_pred)),
        'avg_inference_ms': float(np.mean(inference_times) * 1000),
        'std_inference_ms': float(np.std(inference_times) * 1000),
    }


def run_scalability_benchmark(model_path='weights/cdr_former.pth'):
    """Run scalability benchmark across qubit counts."""
    print("=" * 70)
    print("CDRFormer Scalability Benchmark")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load model
    model = CDRFormer(noise_context_dim=5).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        print(f"Loaded model from: {model_path}")
    else:
        print(f"Warning: No model at {model_path}")
    model.eval()
    
    # Test scales - pushing to 20 qubits
    qubit_counts = [5, 8, 10, 12, 15, 18, 20]
    results = {}
    
    print("\nRunning scalability tests...")
    print("-" * 60)
    
    for n_qubits in qubit_counts:
        print(f"\nTesting {n_qubits} qubits...")
        result = benchmark_scale(model, n_qubits, n_circuits=20)
        
        if result:
            results[n_qubits] = result
            print(f"  Win Rate: {result['win_rate']*100:.1f}%")
            print(f"  MAE Noisy: {result['mae_noisy']:.4f}")
            print(f"  MAE Ours: {result['mae_cdrformer']:.4f}")
            print(f"  Inference: {result['avg_inference_ms']:.1f}ms ± {result['std_inference_ms']:.1f}ms")
        else:
            print(f"  Failed (statevector too large)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SCALABILITY BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'Qubits':>8} {'Win Rate':>12} {'MAE Noisy':>12} {'MAE Ours':>12} {'Inference':>15}")
    print("-" * 65)
    
    for n_qubits, data in results.items():
        print(f"{n_qubits:>8} {data['win_rate']*100:>11.1f}% {data['mae_noisy']:>12.4f} {data['mae_cdrformer']:>12.4f} {data['avg_inference_ms']:>12.1f}ms")
    
    # Save results
    results_path = "assets/scalability_benchmark.json"
    output = {
        'results': {str(k): v for k, v in results.items()},
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'qubit_counts': qubit_counts
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n📊 Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_scalability_benchmark()
