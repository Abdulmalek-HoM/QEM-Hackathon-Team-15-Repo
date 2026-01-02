
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models.qem_former import QEMFormer
import os
import glob

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
    print("--- Starting QEM-Former Training ---")
    
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
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Setup Model
    model = QEMFormer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            # batch.global_attr: [Batch, NoiseContext]
            # We need to reshape if needed, but PyG batches usually concat.
            # Our global_attr is in data object.
            # If DataLoader collates correctly, batch.global_attr should be [Batch, 3]
            
            # Check shape
            if batch.global_attr.dim() == 1: 
                 # Sometimes PyG collates into [Batch*Features]. We want [Batch, Features]
                 # Wait, our Data object has global_attr as tensor. PyG collates by concatenating on dim 0.
                 # If global_attr is [3], then batch will be [Batch*3]. 
                 # We need to ensure it's [Batch, 3].
                 # In data_gen, we did: data.global_attr = tensor(..., dtype=float)
                 # If we make it [1, 3] in data_gen, it stacks to [Batch, 3].
                 pass
            
            # Let's fix dimension in forward if needed or assume data_gen did it right.
            # In data_gen: data.global_attr = torch.tensor(global_features, dtype=torch.float) -> shape [3]
            # Collation -> [Batch * 3].
            # We need to reshape to [Batch, 3].
            # Batch size is batch.num_graphs
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
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # print("  Model saved.")
            
    print(f"Training Complete. Best Val Loss: {best_val_loss:.6f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
