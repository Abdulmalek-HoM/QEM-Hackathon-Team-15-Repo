"""
Cross-Architecture Validation: Does Data > Architecture?
=========================================================
Trains MLP, GCN, and CDRFormer on the SAME CDR+Twirling data
to prove our key thesis: data composition matters more than model choice.

If all architectures achieve high win rates on the same data,
then the DATA is the primary contributor, not the architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, sys, json, glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, TransformerConv
from models.cdr_former import CDRFormer

# ============================================================
# BASELINE MODEL 1: Simple MLP (No Graph Structure)
# ============================================================
class MLPBaseline(nn.Module):
    """
    Baseline MLP that ignores circuit topology completely.
    Takes only global features (noisy EV, n_qubits, depth, noise_scale, ZZ).
    If data matters more than architecture, even this should work reasonably.
    """
    def __init__(self, noise_context_dim=5):
        super().__init__()
        self.dim = noise_context_dim
        self.net = nn.Sequential(
            nn.Linear(noise_context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index, batch, global_attr):
        # PyG concatenates global_attr as 1D; reshape to [batch_size, dim]
        ga = global_attr.view(-1, self.dim)
        return self.net(ga)


# ============================================================
# BASELINE MODEL 2: GCN (Graph structure, no Transformer)
# ============================================================
class GCNBaseline(nn.Module):
    """
    GCN baseline using standard message-passing (no attention).
    Uses graph structure but simpler than TransformerConv.
    """
    def __init__(self, num_gate_types=16, node_dim=64, noise_context_dim=5):
        super().__init__()
        self.ctx_dim = noise_context_dim
        self.node_embedding = nn.Embedding(num_gate_types, node_dim)
        self.param_encoder = nn.Linear(1, node_dim)
        
        self.conv1 = GCNConv(node_dim, node_dim)
        self.conv2 = GCNConv(node_dim, node_dim)
        
        self.context_encoder = nn.Linear(noise_context_dim, node_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(node_dim + node_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index, batch, global_attr):
        gate_ids = x[:, 0].long()
        params = x[:, 1].unsqueeze(1)
        
        h = self.node_embedding(gate_ids) + self.param_encoder(params)
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        graph_embed = global_mean_pool(h, batch)
        # PyG concatenates global_attr as 1D; reshape
        ga = global_attr.view(-1, self.ctx_dim)
        context_embed = F.relu(self.context_encoder(ga))
        combined = torch.cat([graph_embed, context_embed], dim=1)
        return self.regressor(combined)


# ============================================================
# Training & Evaluation
# ============================================================
DATASET_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 60  # Enough for fair comparison
LR = 0.001

def load_data():
    """Load the main CDR+Twirling dataset (with 35% QAOA)."""
    # Use the mixed_35 dataset (our best composition)
    target = os.path.join(DATASET_DIR, "train_data_mixed_35.pt")
    if os.path.exists(target):
        print(f"Loading {target}...")
        return torch.load(target, weights_only=False)
    
    # Fallback: load all chunks + mixed
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "train_data_chunk_*.pt")))
    mixed = os.path.join(DATASET_DIR, "train_data_mixed_0.pt")
    if os.path.exists(mixed):
        files.append(mixed)
    
    all_data = []
    for f in files:
        if 'chunk_999' in f:
            continue
        print(f"Loading {f}...")
        all_data.extend(torch.load(f, weights_only=False))
    
    return all_data


def train_and_evaluate(model, model_name, train_loader, test_loader, device):
    """Train model and return final metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            ga = batch.global_attr.view(-1, 5)
            pred = model(batch.x, batch.edge_index, batch.batch, ga).squeeze()
            loss = criterion(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        
        train_loss /= len(train_loader.dataset)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                ga = batch.global_attr.view(-1, 5)
                pred = model(batch.x, batch.edge_index, batch.batch, ga).squeeze()
                loss = criterion(pred, batch.y)
                val_loss += loss.item() * batch.num_graphs
        
        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")
    
    # Load best weights
    model.load_state_dict(best_state)
    model.eval()
    
    # Calculate win rates on test set
    wins = {'qaoa': 0, 'variational': 0, 'clifford': 0}
    total = {'qaoa': 0, 'variational': 0, 'clifford': 0}
    mae_model = {'qaoa': [], 'variational': [], 'clifford': []}
    mae_noisy = {'qaoa': [], 'variational': [], 'clifford': []}
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            ga = batch.global_attr.view(-1, 5)
            pred = model(batch.x, batch.edge_index, batch.batch, ga).squeeze()
            
            # global_attr already reshaped as ga
            
            for i in range(batch.num_graphs):
                y_ideal = batch.y[i].item() if batch.y.dim() > 0 else batch.y.item()
                y_pred = pred[i].item() if pred.dim() > 0 else pred.item()
                y_noisy = ga[i, 0].item()
                
                # Determine circuit type heuristic
                if abs(abs(y_ideal) - 1.0) < 0.01 or abs(y_ideal) < 0.01:
                    ctype = 'clifford'
                elif abs(y_ideal) < 0.3:
                    ctype = 'qaoa'
                else:
                    ctype = 'variational'
                
                err_model = abs(y_pred - y_ideal)
                err_noisy = abs(y_noisy - y_ideal)
                
                mae_model[ctype].append(err_model)
                mae_noisy[ctype].append(err_noisy)
                total[ctype] += 1
                if err_model < err_noisy:
                    wins[ctype] += 1
    
    results = {
        'model': model_name,
        'best_val_loss': best_val_loss,
        'win_rates': {},
        'mae_model': {},
        'mae_noisy': {}
    }
    
    for ctype in ['qaoa', 'variational', 'clifford']:
        if total[ctype] > 0:
            wr = wins[ctype] / total[ctype] * 100
            mae_m = np.mean(mae_model[ctype]) if mae_model[ctype] else 0
            mae_n = np.mean(mae_noisy[ctype]) if mae_noisy[ctype] else 0
            results['win_rates'][ctype] = wr
            results['mae_model'][ctype] = mae_m
            results['mae_noisy'][ctype] = mae_n
            print(f"  {ctype:12s} | Win Rate: {wr:5.1f}% | MAE Model: {mae_m:.4f} | MAE Noisy: {mae_n:.4f} ({total[ctype]} samples)")
    
    # Overall win rate
    total_wins = sum(wins.values())
    total_samples = sum(total.values())
    overall_wr = total_wins / total_samples * 100 if total_samples > 0 else 0
    results['win_rates']['overall'] = overall_wr
    print(f"  {'OVERALL':12s} | Win Rate: {overall_wr:5.1f}%")
    
    return results


def main():
    print("=" * 70)
    print("Cross-Architecture Validation: Does Data > Architecture?")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    dataset = load_data()
    print(f"Total samples: {len(dataset)}")
    
    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    split = int(0.8 * len(dataset))
    train_data = [dataset[i] for i in indices[:split]]
    test_data = [dataset[i] for i in indices[split:]]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Detect dimensions
    sample = train_data[0]
    global_dim = sample.global_attr.numel()
    print(f"Global feature dim: {global_dim}")
    
    # ============================================================
    # Train 3 architectures on SAME data
    # ============================================================
    all_results = []
    
    # 1. MLP (no graph structure at all)
    mlp = MLPBaseline(noise_context_dim=global_dim)
    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"\nMLP parameters: {mlp_params:,}")
    r1 = train_and_evaluate(mlp, "MLP (No Graph)", train_loader, test_loader, device)
    r1['num_params'] = mlp_params
    all_results.append(r1)
    
    # 2. GCN (graph structure, no attention)
    gcn = GCNBaseline(noise_context_dim=global_dim)
    gcn_params = sum(p.numel() for p in gcn.parameters())
    print(f"\nGCN parameters: {gcn_params:,}")
    r2 = train_and_evaluate(gcn, "GCN (Message Passing)", train_loader, test_loader, device)
    r2['num_params'] = gcn_params
    all_results.append(r2)
    
    # 3. CDRFormer (Graph Transformer — our full model)
    cdrformer = CDRFormer(noise_context_dim=global_dim)
    cdr_params = sum(p.numel() for p in cdrformer.parameters())
    print(f"\nCDRFormer parameters: {cdr_params:,}")
    r3 = train_and_evaluate(cdrformer, "CDRFormer (Graph Transformer)", train_loader, test_loader, device)
    r3['num_params'] = cdr_params
    all_results.append(r3)
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Params':>10} {'QAOA Win%':>10} {'Var Win%':>10} {'Overall':>10} {'Val Loss':>10}")
    print("-" * 80)
    
    for r in all_results:
        qaoa = r['win_rates'].get('qaoa', 0)
        var = r['win_rates'].get('variational', 0)
        overall = r['win_rates'].get('overall', 0)
        print(f"{r['model']:<30} {r['num_params']:>10,} {qaoa:>9.1f}% {var:>9.1f}% {overall:>9.1f}% {r['best_val_loss']:>10.6f}")
    
    print()
    print("KEY FINDING:")
    qaoa_rates = [r['win_rates'].get('qaoa', 0) for r in all_results]
    if min(qaoa_rates) > 70:
        print("✅ ALL architectures achieve high QAOA win rates on the SAME data.")
        print("   → DATA COMPOSITION is the primary contributor, NOT architecture.")
    else:
        print("⚠️  Architecture differences observed. CDRFormer may have architectural advantage.")
    
    # Save
    save_path = "assets/cross_architecture_results.json"
    os.makedirs("assets", exist_ok=True)
    
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        return obj
    
    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
