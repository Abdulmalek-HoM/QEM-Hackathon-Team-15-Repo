
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models.qem_former import QEMFormer
import os
import glob
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Configuration
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "qem_former.pth"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100

def load_data():
    """Loads all .pt files from dataset directory"""
    files = glob.glob(os.path.join(DATASET_DIR, "*.pt"))
    if not files:
        raise ValueError("No dataset found! Run data_gen_advanced.py first.")
    
    all_data = []
    for f in files:
        print(f"Loading {f}...")
        data_chunk = torch.load(f, weights_only=False)
        all_data.extend(data_chunk)
        
    return all_data

def train():
    print("=" * 60)
    print("QEM-Former Training Pipeline")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    try:
        dataset = load_data()
    except ValueError as e:
        print(e)
        return
        
    print(f"Total Samples: {len(dataset)}")
    
    # Split Train/Test (Simple 80/20)
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Setup Model
    # Detect global_attr dimension from first sample
    sample = train_data[0]
    global_dim = sample.global_attr.numel()
    print(f"Detected global feature dimension: {global_dim}")
    
    model = QEMFormer(noise_context_dim=global_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'timestamp': datetime.now().isoformat()
    }
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    print("\n--- Training Started ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Reshape global attributes
            g_attr = batch.global_attr.reshape(batch.num_graphs, -1)
            
            output = model(batch.x, batch.edge_index, batch.batch, g_attr)
            
            loss = criterion(output.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                g_attr = batch.global_attr.reshape(batch.num_graphs, -1)
                pred = model(batch.x, batch.edge_index, batch.batch, g_attr)
                loss = criterion(pred.squeeze(), batch.y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        metrics['train_loss'].append(avg_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['learning_rate'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {avg_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            metrics['best_epoch'] = epoch + 1
            metrics['best_val_loss'] = best_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved!")
            
    print("\n" + "=" * 60)
    print(f"Training Complete. Best Val Loss: {best_val_loss:.6f} (Epoch {metrics['best_epoch']})")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 4. Save Training Metrics
    metrics_path = "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # 5. Generate Training Visualization
    plot_training_curves(metrics)
    
    return metrics

def plot_training_curves(metrics):
    """Generate and save training visualization plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Plot 1: Loss Curves
    ax1 = axes[0]
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=metrics['best_epoch'], color='g', linestyle='--', alpha=0.7, label=f'Best (Epoch {metrics["best_epoch"]})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('QEM-Former Training Curves', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Learning Rate
    ax2 = axes[1]
    ax2.plot(epochs, metrics['learning_rate'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {plot_path}")

if __name__ == "__main__":
    train()
