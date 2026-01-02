
import numpy as np
import torch
import torch.nn as nn
from qiskit_aer import AerSimulator
from qiskit import transpile
import mitiq
from mitiq import zne
import utils
from data_gen_advanced import QEMGraphBuilder, generate_advanced_dataset, DATASET_DIR
from models.qem_former import QEMFormer
import os
from torch_geometric.loader import DataLoader

def benchmark_models(model_path="qem_former.pth"):
    """
    Benchmarks the trained QEM-Former against Noisy Baseline and Mitiq ZNE.
    """
    print("--- Starting Benchmark Suite ---")
    
    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QEMFormer().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded QEM-Former weights.")
        model.eval()
    else:
        print("WARNING: Model weights not found. Using random weights (Benchmarking untrained model).")
    
    # 2. Setup Simulators
    sim_ideal = AerSimulator(method='stabilizer')
    noise_model = utils.build_noise_model(scale=1.5)
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    # 3. Generate Evaluation Circuits (Random Clifford or QAOA)
    # Let's use 20 evaluation circuits
    n_eval = 50
    circuits = []
    
    print(f"Generating {n_eval} evaluation circuits...")
    for _ in range(n_eval):
        qc, _ = utils.create_random_clifford_circuit(5, 20) # 5 qubits, depth 20
        qc.measure_all()
        circuits.append(qc)
        
    results = {
        'noisy_mae': [],
        'zne_mae': [],
        'qem_mae': [],
        'improvement_ratio': []
    }
    
    builder = QEMGraphBuilder()
    
    def executor(circuit):
        """Executor for Mitiq ZNE"""
        # Remove measurements for Mitiq if needed, but Mitiq usually handles it.
        # Mitiq expects a function that takes a circuit and returns a float (expectation).
        # We need counts.
        t_qc = transpile(circuit, sim_noisy)
        res = sim_noisy.run(t_qc, shots=1000).result().get_counts()
        # Expectation <Z0>
        z0 = 0
        total = 0
        for b, c in res.items():
            val = 1 if b[-1] == '0' else -1
            z0 += val * c
            total += c
        return z0 / total if total > 0 else 0

    for i, qc in enumerate(circuits):
        # A. Ideal
        qc_sim = qc.copy()
        res_ideal = sim_ideal.run(qc_sim, shots=1000).result().get_counts()
        z0_ideal = 0
        total = 0
        for b, c in res_ideal.items():
            val = 1 if b[-1] == '0' else -1
            z0_ideal += val * c
            total += c
        val_ideal = z0_ideal / total
        
        # B. Raw Noisy
        val_noisy = executor(qc)
        
        # C. Mitigated: ZNE
        # Simple Linear ZNE with scale factors 1, 3, 5
        fac = zne.inference.LinearFactory(scale_factors=[1.0, 3.0]) 
        # Note: Scaling noise in Aer is tricky without `zne.scaling.fold_gates_at_random`.
        # Mitiq automatically scales the circuit structure (folding).
        val_zne = zne.execute_with_zne(qc, executor, factory=fac, scale_noise=zne.scaling.fold_gates_at_random)
        
        # D. Mitigated: QEM-Former
        # Build Graph
        # Global attributes: [Noisy_Val, n_qubits, depth]
        # We use the raw noisy value as context
        global_attr = [val_noisy, 5.0, 20.0]
        graph = builder.circuit_to_graph(qc, global_features=global_attr).to(device)
        
        with torch.no_grad():
            # Create batch of size 1
            batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(device)
            pred = model(graph.x, graph.edge_index, batch, graph.global_attr.unsqueeze(0))
            val_qem = pred.item()
            
        # Metrics
        err_noisy = abs(val_noisy - val_ideal)
        err_zne = abs(val_zne - val_ideal)
        err_qem = abs(val_qem - val_ideal)
        
        results['noisy_mae'].append(err_noisy)
        results['zne_mae'].append(err_zne)
        results['qem_mae'].append(err_qem)
        
        # IR: abs(Noisy - Ideal) / abs(Mitigated - Ideal)
        # Avoid div by zero
        ir = err_noisy / (err_qem + 1e-9)
        results['improvement_ratio'].append(ir)
        
        print(f"Circ {i}: Ideal={val_ideal:.3f} | Noisy={val_noisy:.3f} (E={err_noisy:.3f}) | ZNE={val_zne:.3f} (E={err_zne:.3f}) | QEM={val_qem:.3f} (E={err_qem:.3f}, IR={ir:.1f})")

    avg_ir = np.mean(results['improvement_ratio'])
    avg_qem_err = np.mean(results['qem_mae'])
    
    print("\n--- Summary ---")
    print(f"Mean Noisy Error: {np.mean(results['noisy_mae']):.4f}")
    print(f"Mean ZNE Error:   {np.mean(results['zne_mae']):.4f}")
    print(f"Mean QEM Error:   {avg_qem_err:.4f}")
    print(f"Mean Improvement Ratio: {avg_ir:.2f}x")
    
    if avg_qem_err < np.mean(results['noisy_mae']):
        print("SUCCESS: QEM Model is improving results!")
    else:
        print("WARNING: QEM Model requires more training.")

if __name__ == "__main__":
    benchmark_models()
