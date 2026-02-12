"""
Data Composition Ablation Study for CDRFormer

This script tests the hypothesis that data composition impacts
QEM performance more than architecture complexity.

Key experiment: Vary QAOA fraction in training data and measure
impact on QAOA win rate.

Expected finding: QAOA win rate scales with QAOA training fraction.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datetime import datetime
import json

from data_gen_advanced import generate_mixed_dataset, QEMGraphBuilder, DATASET_DIR
from models.cdr_former import CDRFormer
import utils
from scripts.benchmark_publication import run_full_benchmark

# ============================================================
# Configuration
# ============================================================

ABLATION_CONFIGS = [
    # (name, clifford_frac, qaoa_frac, variational_frac)
    ("low_qaoa", 0.80, 0.08, 0.12),      # Original hackathon setting
    ("medium_qaoa", 0.60, 0.20, 0.20),   # Balanced
    ("high_qaoa", 0.40, 0.35, 0.25),     # Post-hackathon (current)
    ("very_high_qaoa", 0.30, 0.50, 0.20), # Experimental
]

N_SAMPLES = 20000  # Per ablation config - increased for statistical significance
N_EPOCHS = 100     # More training for larger dataset
BATCH_SIZE = 64    # Larger batch for efficiency
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Training Function
# ============================================================

def train_model(dataloader, n_epochs=N_EPOCHS, verbose=True):
    """Train a CDRFormer model and return it."""
    model = CDRFormer(noise_context_dim=5).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # global_attr is concatenated by DataLoader - reshape to (batch_size, 5)
            batch_size = batch.y.size(0)
            global_attr = batch.global_attr.view(batch_size, -1)
            
            pred = model(batch.x, batch.edge_index, batch.batch, global_attr)
            loss = criterion(pred.squeeze(), batch.y.squeeze())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/n_batches:.4f}")
    
    return model


# ============================================================
# Quick Benchmark Function
# ============================================================

def quick_benchmark(model, n_circuits=20, seed=42):
    """
    Quick benchmark on QAOA circuits only.
    Returns win rate vs noisy baseline.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    
    noise_model = utils.build_noise_model(scale=1.5)
    sim_noisy = AerSimulator(noise_model=noise_model)
    builder = QEMGraphBuilder()
    
    def executor(circuit):
        t_qc = transpile(circuit, sim_noisy)
        res = sim_noisy.run(t_qc, shots=2000).result().get_counts()
        z0 = 0
        total = 0
        for b, c in res.items():
            val = 1 if b[-1] == '0' else -1
            z0 += val * c
            total += c
        return z0 / total if total > 0 else 0
    
    wins = 0
    model.eval()
    
    for _ in range(n_circuits):
        qc, _ = utils.create_qaoa_circuit(5, p=2)
        qc.measure_all()
        
        # Ideal
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        probs = sv.probabilities()
        val_ideal = sum((1 if (i >> 0) & 1 == 0 else -1) * p for i, p in enumerate(probs))
        
        # Noisy
        val_noisy = executor(qc)
        
        # CDRFormer prediction
        global_attr = [val_noisy, 0.0, 5.0, 20.0, 1.5]
        graph = builder.circuit_to_graph(qc, global_features=global_attr).to(DEVICE)
        
        with torch.no_grad():
            batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(DEVICE)
            pred = model(graph.x, graph.edge_index, batch, graph.global_attr.unsqueeze(0))
            val_pred = pred.item()
        
        err_noisy = abs(val_noisy - val_ideal)
        err_pred = abs(val_pred - val_ideal)
        
        if err_pred < err_noisy:
            wins += 1
    
    return wins / n_circuits


# ============================================================
# Main Ablation Study
# ============================================================

def run_ablation_study():
    """Run full data composition ablation study."""
    print("=" * 70)
    print("CDRFormer Data Composition Ablation Study")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print()
    
    results = {}
    
    for config_name, clf_frac, qaoa_frac, var_frac in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  Clifford: {clf_frac*100:.0f}%, QAOA: {qaoa_frac*100:.0f}%, Variational: {var_frac*100:.0f}%")
        print(f"{'='*60}")
        
        # Step 1: Generate dataset
        print("\n[1/3] Generating dataset...")
        chunk_id = int(qaoa_frac * 100)  # Unique ID per config
        generate_mixed_dataset(
            n_samples=N_SAMPLES,
            min_qubits=4,
            max_qubits=6,
            noise_scale=1.5,
            chunk_id=chunk_id,
            clifford_frac=clf_frac,
            qaoa_frac=qaoa_frac,
            variational_frac=var_frac
        )
        
        # Step 2: Load and train
        print("\n[2/3] Training model...")
        dataset_path = os.path.join(DATASET_DIR, f"train_data_mixed_{chunk_id}.pt")
        dataset = torch.load(dataset_path, weights_only=False)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = train_model(dataloader, n_epochs=N_EPOCHS, verbose=True)
        
        # Step 3: Benchmark
        print("\n[3/3] Benchmarking on QAOA circuits...")
        qaoa_win_rate = quick_benchmark(model, n_circuits=30, seed=42)
        print(f"  QAOA Win Rate: {qaoa_win_rate*100:.1f}%")
        
        # Store results
        results[config_name] = {
            'clifford_frac': clf_frac,
            'qaoa_frac': qaoa_frac,
            'variational_frac': var_frac,
            'qaoa_win_rate': qaoa_win_rate,
        }
        
        # Save model
        model_path = f"weights/cdr_former_qaoa{int(qaoa_frac*100)}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved to: {model_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<20} {'QAOA %':>10} {'Win Rate':>12}")
    print("-" * 45)
    for config_name, data in results.items():
        print(f"{config_name:<20} {data['qaoa_frac']*100:>9.0f}% {data['qaoa_win_rate']*100:>11.1f}%")
    
    # Save results
    results_path = "assets/data_composition_ablation.json"
    results['timestamp'] = datetime.now().isoformat()
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_ablation_study()
