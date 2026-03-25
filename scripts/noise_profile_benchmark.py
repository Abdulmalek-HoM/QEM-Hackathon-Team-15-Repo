"""
QEM-Bench Compatible Noise Profile Benchmark

Tests CDRFormer across noise profiles matching QEM-Bench:
1. Incoherent noise (depolarizing) - our default
2. FakeBackend noise (FakeBrisbane, FakeKyoto) - IBM-like noise
3. Coherent noise (over-rotation errors) - systematic bias

This validates CDRFormer generalizes to diverse noise conditions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from datetime import datetime
import json

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, coherent_unitary_error
from qiskit.quantum_info import Statevector

# Try to import FakeProviders (Qiskit-IBM-Runtime)
try:
    from qiskit_ibm_runtime.fake_provider import FakeHanoiV2, FakeCairoV2
    HAS_FAKE_BACKENDS = True
except ImportError:
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
        HAS_FAKE_BACKENDS = False  # Use generic instead
    except ImportError:
        HAS_FAKE_BACKENDS = False
        print("Warning: FakeBackends not available.")

from data_gen_advanced import QEMGraphBuilder
from models.cdr_former import CDRFormer
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SHOTS = 2000


# ============================================================
# Noise Model Builders
# ============================================================

def build_incoherent_noise(error_rate=0.01):
    """Incoherent (depolarizing) noise - our default."""
    noise_model = NoiseModel()
    
    # Single-qubit gates
    error_1q = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['x', 'y', 'z', 'h', 's', 't', 'sx', 'rz'])
    
    # Two-qubit gates
    error_2q = depolarizing_error(error_rate * 10, 2)  # 2Q gates are noisier
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'ecr'])
    
    return noise_model


def build_coherent_noise(theta_error=0.05):
    """
    Coherent (over-rotation) noise.
    Each gate has a small systematic rotation error.
    This tests if CDRFormer can handle non-stochastic noise.
    """
    noise_model = NoiseModel()
    
    # Over-rotation on single-qubit gates (small systematic error)
    # Creates a unitary error: RZ(theta_error)
    rz_error = np.array([
        [np.exp(-1j * theta_error / 2), 0],
        [0, np.exp(1j * theta_error / 2)]
    ])
    coherent_err = coherent_unitary_error(rz_error)
    noise_model.add_all_qubit_quantum_error(coherent_err, ['x', 'y', 'z', 'h', 'sx', 'rz'])
    
    return noise_model


def build_combined_noise(depol_rate=0.008, theta_error=0.03):
    """Combined incoherent + coherent noise (most realistic)."""
    noise_model = NoiseModel()
    
    # Incoherent
    error_1q = depolarizing_error(depol_rate, 1)
    error_2q = depolarizing_error(depol_rate * 10, 2)
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['x', 'y', 'z', 'h', 's', 't', 'sx', 'rz'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'ecr'])
    
    # Coherent overlay
    rz_error = np.array([
        [np.exp(-1j * theta_error / 2), 0],
        [0, np.exp(1j * theta_error / 2)]
    ])
    coherent_err = coherent_unitary_error(rz_error)
    noise_model.add_all_qubit_quantum_error(coherent_err, ['x', 'y', 'z', 'h'])
    
    return noise_model


def get_fake_backend_noise(backend_name='hanoi'):
    """Get noise model from IBM FakeBackend."""
    if not HAS_FAKE_BACKENDS:
        print(f"FakeBackend {backend_name} not available, using incoherent fallback")
        return build_incoherent_noise()
    
    try:
        if backend_name == 'hanoi':
            backend = FakeHanoiV2()
        elif backend_name == 'cairo':
            backend = FakeCairoV2()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        return NoiseModel.from_backend(backend)
    except Exception as e:
        print(f"Error loading {backend_name}: {e}")
        return build_incoherent_noise()


# ============================================================
# Benchmark Function
# ============================================================

def benchmark_noise_profile(noise_model, noise_name, model, n_circuits=30, seed=42):
    """
    Benchmark CDRFormer on a specific noise profile.
    Returns win rate vs noisy baseline on QAOA circuits.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sim_noisy = AerSimulator(noise_model=noise_model)
    builder = QEMGraphBuilder()
    model.eval()
    
    wins = 0
    errors_noisy = []
    errors_pred = []
    
    for i in range(n_circuits):
        # Generate QAOA circuit
        qc, _ = utils.create_qaoa_circuit(5, p=2)
        qc.measure_all()
        
        # Ideal value via statevector
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        probs = sv.probabilities()
        val_ideal = sum((1 if (idx >> 0) & 1 == 0 else -1) * p for idx, p in enumerate(probs))
        
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
        
        # CDRFormer prediction
        global_attr = [val_noisy, 0.0, 5.0, 20.0, 1.5]  # Use standard params
        graph = builder.circuit_to_graph(qc, global_features=global_attr).to(DEVICE)
        
        with torch.no_grad():
            batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(DEVICE)
            pred = model(graph.x, graph.edge_index, batch, graph.global_attr.unsqueeze(0))
            val_pred = pred.item()
        
        err_noisy = abs(val_noisy - val_ideal)
        err_pred = abs(val_pred - val_ideal)
        
        errors_noisy.append(err_noisy)
        errors_pred.append(err_pred)
        
        if err_pred < err_noisy:
            wins += 1
    
    win_rate = wins / len(errors_noisy) if errors_noisy else 0
    mae_noisy = np.mean(errors_noisy) if errors_noisy else 0
    mae_pred = np.mean(errors_pred) if errors_pred else 0
    
    return {
        'noise_profile': noise_name,
        'win_rate': win_rate,
        'mae_noisy': mae_noisy,
        'mae_cdrformer': mae_pred,
        'n_circuits': len(errors_noisy)
    }


# ============================================================
# Main Benchmark Suite
# ============================================================

def run_noise_profile_benchmark(model_path='weights/cdr_former.pth'):
    """Run benchmark across all noise profiles."""
    print("=" * 70)
    print("CDRFormer Noise Profile Benchmark")
    print("QEM-Bench Compatible Testing")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Load model
    model = CDRFormer(noise_context_dim=5).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        print(f"Loaded model from: {model_path}")
    else:
        print(f"Warning: No model at {model_path}, using random weights")
    model.eval()
    
    # Define noise profiles
    noise_profiles = [
        ('Incoherent (Depolarizing)', build_incoherent_noise(error_rate=0.01)),
        ('Coherent (Over-rotation)', build_coherent_noise(theta_error=0.05)),
        ('Combined (Incoh+Coh)', build_combined_noise(depol_rate=0.008, theta_error=0.03)),
    ]
    
    # Add FakeBackends if available
    if HAS_FAKE_BACKENDS:
        noise_profiles.extend([
            ('FakeHanoi (27Q IBM)', get_fake_backend_noise('hanoi')),
            ('FakeCairo (27Q IBM)', get_fake_backend_noise('cairo')),
        ])
    
    # Run benchmarks
    results = {}
    print("\nRunning benchmarks...")
    print("-" * 60)
    
    for noise_name, noise_model in noise_profiles:
        print(f"\nTesting: {noise_name}...")
        result = benchmark_noise_profile(noise_model, noise_name, model, n_circuits=30)
        results[noise_name] = result
        print(f"  Win Rate: {result['win_rate']*100:.1f}%")
        print(f"  MAE Noisy: {result['mae_noisy']:.4f}")
        print(f"  MAE CDRFormer: {result['mae_cdrformer']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("NOISE PROFILE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'Noise Profile':<25} {'Win Rate':>12} {'MAE Noisy':>12} {'MAE Ours':>12}")
    print("-" * 65)
    
    for name, data in results.items():
        print(f"{name:<25} {data['win_rate']*100:>11.1f}% {data['mae_noisy']:>12.4f} {data['mae_cdrformer']:>12.4f}")
    
    # Save results
    results_path = "assets/noise_profile_benchmark.json"
    output = {
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'fake_backends_available': HAS_FAKE_BACKENDS
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n📊 Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_noise_profile_benchmark()
