"""
Pauli Twirling Ablation Study

Tests the hypothesis that Pauli Twirling improves QEM training by
converting coherent noise to stochastic (Markovian) noise.

Experiment:
1. Generate dataset WITHOUT Pauli Twirling
2. Train model on non-twirled data
3. Compare performance to twirled model
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from datetime import datetime
import json

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from data_gen_advanced import QEMGraphBuilder, calculate_z0_expectation, DATASET_DIR
from models.cdr_former import CDRFormer
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLES = 2000
N_EPOCHS = 80
BATCH_SIZE = 32


def generate_dataset_no_twirling(n_samples=N_SAMPLES, noise_scale=1.5, seed=42):
    """
    Generate dataset WITHOUT Pauli Twirling.
    This preserves coherent noise structure.
    """
    np.random.seed(seed)
    
    builder = QEMGraphBuilder()
    noise_model = utils.build_noise_model(scale=noise_scale)
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    data_list = []
    
    print(f"Generating {n_samples} samples WITHOUT Pauli Twirling...")
    
    for i in range(n_samples):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        # Mix of circuit types (40% Clifford, 35% QAOA, 25% Variational)
        circuit_type = np.random.choice(['clifford', 'qaoa', 'variational'], p=[0.4, 0.35, 0.25])
        n_q = np.random.randint(4, 7)
        
        if circuit_type == 'clifford':
            depth = np.random.randint(n_q, n_q * 3)
            qc, _ = utils.create_random_clifford_circuit(n_q, depth)
        elif circuit_type == 'qaoa':
            p = np.random.randint(1, 4)
            qc, _ = utils.create_qaoa_circuit(n_q, p=p)
            depth = p * 3 * n_q
        else:
            depth = np.random.randint(3, 8)
            qc, _ = utils.create_variational_circuit(n_q, depth)
        
        qc.measure_all()
        
        # Ideal value via statevector
        try:
            qc_no_meas = qc.remove_final_measurements(inplace=False)
            sv = Statevector.from_instruction(qc_no_meas)
            probs = sv.probabilities()
            val_ideal = sum((1 if (idx >> 0) & 1 == 0 else -1) * p for idx, p in enumerate(probs))
        except:
            continue
        
        # Noisy execution WITHOUT twirling
        t_qc = transpile(qc, basis_gates=sim_noisy.operation_names)
        counts = sim_noisy.run(t_qc, shots=2000).result().get_counts()
        val_noisy = calculate_z0_expectation(counts)
        
        # Build graph
        global_feats = [val_noisy, 0.0, float(n_q), float(depth), noise_scale]
        graph_data = builder.circuit_to_graph(qc, global_features=global_feats)
        graph_data.y = torch.tensor([val_ideal], dtype=torch.float)
        
        data_list.append(graph_data)
    
    # Save
    save_path = os.path.join(DATASET_DIR, "train_data_no_twirling.pt")
    torch.save(data_list, save_path)
    print(f"Saved {len(data_list)} samples to {save_path}")
    
    return data_list


def train_model(dataloader, n_epochs=N_EPOCHS):
    """Train a CDRFormer model."""
    model = CDRFormer(noise_context_dim=5).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            batch_size = batch.y.size(0)
            global_attr = batch.global_attr.view(batch_size, -1)
            
            pred = model(batch.x, batch.edge_index, batch.batch, global_attr)
            loss = criterion(pred.squeeze(), batch.y.squeeze())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/n_batches:.4f}")
    
    return model


def benchmark_model(model, n_circuits=30, seed=42):
    """Benchmark model on QAOA circuits."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    noise_model = utils.build_noise_model(scale=1.5)
    sim_noisy = AerSimulator(noise_model=noise_model)
    builder = QEMGraphBuilder()
    model.eval()
    
    wins = 0
    
    for _ in range(n_circuits):
        qc, _ = utils.create_qaoa_circuit(5, p=2)
        qc.measure_all()
        
        # Ideal
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        probs = sv.probabilities()
        val_ideal = sum((1 if (idx >> 0) & 1 == 0 else -1) * p for idx, p in enumerate(probs))
        
        # Noisy
        t_qc = transpile(qc, basis_gates=sim_noisy.operation_names)
        counts = sim_noisy.run(t_qc, shots=2000).result().get_counts()
        val_noisy = calculate_z0_expectation(counts)
        
        # Predict
        global_attr = [val_noisy, 0.0, 5.0, 20.0, 1.5]
        graph = builder.circuit_to_graph(qc, global_features=global_attr).to(DEVICE)
        
        with torch.no_grad():
            batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(DEVICE)
            pred = model(graph.x, graph.edge_index, batch, graph.global_attr.unsqueeze(0))
            val_pred = pred.item()
        
        if abs(val_pred - val_ideal) < abs(val_noisy - val_ideal):
            wins += 1
    
    return wins / n_circuits


def run_twirling_ablation():
    """Run full Pauli Twirling ablation study."""
    print("=" * 70)
    print("Pauli Twirling Ablation Study")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    results = {}
    
    # --- Model WITHOUT Twirling ---
    print("\n[1/2] Training model WITHOUT Pauli Twirling...")
    
    # Generate data without twirling
    dataset_no_twirl = generate_dataset_no_twirling(n_samples=N_SAMPLES)
    dataloader_no_twirl = DataLoader(dataset_no_twirl, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train
    model_no_twirl = train_model(dataloader_no_twirl, n_epochs=N_EPOCHS)
    
    # Benchmark
    win_rate_no_twirl = benchmark_model(model_no_twirl, n_circuits=30)
    print(f"  QAOA Win Rate (No Twirling): {win_rate_no_twirl*100:.1f}%")
    
    results['no_twirling'] = {
        'win_rate': win_rate_no_twirl,
        'n_samples': N_SAMPLES,
        'n_epochs': N_EPOCHS
    }
    
    # Save model
    torch.save(model_no_twirl.state_dict(), "weights/cdr_former_no_twirling.pth")
    
    # --- Model WITH Twirling (existing) ---
    print("\n[2/2] Loading model WITH Pauli Twirling (existing)...")
    
    model_with_twirl = CDRFormer(noise_context_dim=5).to(DEVICE)
    model_with_twirl.load_state_dict(torch.load("weights/cdr_former.pth", map_location=DEVICE, weights_only=False))
    
    win_rate_with_twirl = benchmark_model(model_with_twirl, n_circuits=30)
    print(f"  QAOA Win Rate (With Twirling): {win_rate_with_twirl*100:.1f}%")
    
    results['with_twirling'] = {
        'win_rate': win_rate_with_twirl,
        'note': 'Loaded from existing trained model'
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("PAULI TWIRLING ABLATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'QAOA Win Rate':>15}")
    print("-" * 45)
    print(f"{'WITHOUT Twirling':<25} {win_rate_no_twirl*100:>14.1f}%")
    print(f"{'WITH Twirling':<25} {win_rate_with_twirl*100:>14.1f}%")
    
    improvement = (win_rate_with_twirl - win_rate_no_twirl) * 100
    print(f"\nTwirling Benefit: {improvement:+.1f} percentage points")
    
    # Save
    results['improvement_pp'] = improvement
    results['timestamp'] = datetime.now().isoformat()
    
    with open("assets/twirling_ablation.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: assets/twirling_ablation.json")
    
    return results


if __name__ == "__main__":
    run_twirling_ablation()
