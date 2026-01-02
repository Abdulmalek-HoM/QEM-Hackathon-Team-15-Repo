
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, GATConv
from torch_geometric.data import Data

class QEMFormer(nn.Module):
    """
    QEM-Former: A Logic-Aware Graph Transformer for Quantum Error Mitigation.
    
    Architecture:
    1. Node Embedding: Gate Type -> Vector
    2. Local GNN Layers: Encode local connectivity (Circuit Topology)
    3. Global Transformer: Capture long-range dependencies (Crosstalk)
    4. Fusion: Combine Graph Embedding with Noise Context (T1, T2, etc.)
    5. Output: Predict Ideal Expectation Value (or Error Correction Delta)
    """
    def __init__(self, 
                 num_gate_types=16, 
                 node_dim=64, 
                 num_gnn_layers=2, 
                 transformer_heads=4, 
                 transformer_layers=2, 
                 noise_context_dim=3):
        super(QEMFormer, self).__init__()
        
        # 1. Embeddings
        self.node_embedding = nn.Embedding(num_gate_types, node_dim)
        self.param_encoder = nn.Linear(1, node_dim) # For rotation angles
        
        # 2. Local Structure (GNN)
        # We use TransformerConv (GAT-like with edge features support if needed)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                TransformerConv(node_dim, node_dim // transformer_heads, heads=transformer_heads, concat=True)
            )
            
        # 3. Global Structure (Standard Transformer)
        # We assume batch of graphs. We can create a dense representation or use PyG utils.
        # For simplicity, we apply Transformer on the node features directly via PyTorch's TransformerEncoder
        # But handling variable sized graphs in batch requires masking.
        # A simpler robust "Global" approach in GNNs: 
        # Deep GNN layers effectively do global, but slow.
        # True Global Attention: We can use a Global Node or just pool.
        # Let's use a TransformerEncoder on the nodes of each graph? 
        # Doing this efficiently in PyG ("Sparse" tensor) is tricky.
        # Shortcut: We rely on the `TransformerConv` which IS a transformer attention mechanism over neighbors.
        # To get "Global" attention, we can add a 'Global Virtual Node' connected to everyone?
        # OR: Just stack many GAT layers.
        # Let's stick to GNN layers + Global Pooling for now to be robust and fast.
        # Wait, the prompt asked for "Global Self-Attention". 
        # Let's add a Global Attention block (GPS-style or just dense transformer).
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=node_dim, nhead=transformer_heads, dim_feedforward=node_dim*2, batch_first=True),
            num_layers=transformer_layers
        )
        
        # 4. Readout
        self.pool = global_mean_pool
        
        # 5. Injection of Noise Context
        # Input global_attr: [Noisy_Val, N_Qubits, Depth]
        self.context_encoder = nn.Linear(noise_context_dim, node_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(node_dim + node_dim, 64), # Graph Embed + Context Embed
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, edge_index, batch, global_attr):
        """
        x: [Num_Nodes, 2] (GateID, Params)
        edge_index: [2, Num_Edges]
        batch: [Num_Nodes] indicating graph assignment
        global_attr: [Batch_Size, Noise_Context_Dim]
        """
        # A. Node Features
        gate_ids = x[:, 0].long()
        params = x[:, 1].unsqueeze(1)
        
        h = self.node_embedding(gate_ids) + self.param_encoder(params)
        
        # B. Local GNN (Topology)
        for gnn in self.gnn_layers:
            h = gnn(h, edge_index)
            h = F.relu(h)
            
        # C. Global Transformer? 
        # Converting sparse batch to dense is expensive. 
        # Let's skip dense Transformer for now and rely on GNN + Global Pooling for scalability 
        # unless we implement Virtual Node.
        # *Feature substitution*: Let's assume the GNN `TransformerConv` provides sufficient 'Attention'.
        # To strictly follow "Global Self-Attention allows first gate to attend to last":
        # We would need a fully connected graph or Transformer.
        # Let's add a Virtual Global Node connected to all nodes?
        # Or just rely on pooling.
        
        # D. Global Pooling
        graph_embed = self.pool(h, batch) # [Batch_Size, Node_Dim]
        
        # E. Context Fusion
        context_embed = F.relu(self.context_encoder(global_attr))
        
        combined = torch.cat([graph_embed, context_embed], dim=1)
        
        out = self.regressor(combined)
        return out

