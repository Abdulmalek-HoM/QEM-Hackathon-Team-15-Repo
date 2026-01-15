# Code Highlights
## Key Code Snippets for Video Presentation

These are the most important code sections to show during the technical demo. Each snippet demonstrates a core concept of the QEM pipeline.

---

## 1. PAULI TWIRLING 
**File:** `data_gen_advanced.py` (Lines 117-220)

This is the most impressive technical contribution - converting coherent errors to stochastic.

```python
def apply_pauli_twirling(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Applies Pauli Twirling to CNOT gates in the circuit.
    Converts coherent errors into stochastic Pauli errors.
    """
    twirled_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    
    for instruction in qc.data:
        op = instruction.operation
        qubits = instruction.qubits
        
        if op.name == 'cx':  # Target CNOTs (dominant error source)
            paulis = ['id', 'x', 'y', 'z']
            
            # Random Pauli selection
            idx_c = np.random.randint(4)
            idx_t = np.random.randint(4)
            
            # PRE Gates
            if pc_gate != 'id': 
                getattr(twirled_qc, pc_gate)(qubits[0])
            if pt_gate != 'id': 
                getattr(twirled_qc, pt_gate)(qubits[1])
            
            # Original CNOT
            twirled_qc.cx(qubits[0], qubits[1])
            
            # POST Gates (from lookup table)
            lookup = {
                (0,0):(0,0), (0,1):(0,1), (0,2):(3,2), (0,3):(3,3),
                (1,0):(1,1), (1,1):(1,0), (1,2):(2,3), (1,3):(2,2),
                (2,0):(2,1), (2,1):(2,0), (2,2):(1,3), (2,3):(1,2),
                (3,0):(3,0), (3,1):(3,1), (3,2):(0,2), (3,3):(0,3)
            }
            # ... apply correction gates
```

**What to say:**
> "We insert random Paulis before each CNOT and carefully chosen Paulis after, preserving the logical operation while randomizing noise."

---

## 2. CIRCUIT-TO-GRAPH CONVERSION
**File:** `data_gen_advanced.py` (Lines 33-114)

This shows how we represent quantum circuits as graphs.

```python
def circuit_to_graph(self, qc: QuantumCircuit, global_features=None) -> Data:
    """Converts a quantum circuit to a Graph Data object."""
    dag = circuit_to_dag(qc)  # Qiskit DAG representation
    node_features = []
    edge_index = [[], []]
    
    # Gate vocabulary
    gate_vocab = {"h": 0, "s": 1, "sdg": 2, "x": 3, "y": 4, "z": 5, 
                  "cx": 7, "rx": 10, "ry": 11, "rz": 12, ...}
    
    # Node features: [gate_id, parameter]
    for node in dag.topological_op_nodes():
        gate_id = gate_vocab.get(node.name, 15)
        param = float(node.op.params[0]) if node.op.params else 0.0
        node_features.append([gate_id, param])
    
    # Edge construction: connect gates on same qubit
    last_node_on_qubit = {q: -1 for q in qc.qubits}
    for current_idx, node in enumerate(dag.topological_op_nodes()):
        for q in node.qargs:
            prev_idx = last_node_on_qubit[q]
            if prev_idx != -1:
                edge_index[0].append(prev_idx)  # Source
                edge_index[1].append(current_idx)  # Target
            last_node_on_qubit[q] = current_idx
    
    return Data(x=torch.tensor(node_features), edge_index=torch.tensor(edge_index))
```

**What to say:**
> "We traverse the circuit DAG. Each gate becomes a node, each qubit wire becomes edges connecting sequential operations."

---

## 3. QEM-FORMER ARCHITECTURE
**File:** `models/qem_former.py` (Lines 8-74)

The complete model definition.

```python
class QEMFormer(nn.Module):
    """
    QEM-Former: A Logic-Aware Graph Transformer for Quantum Error Mitigation.
    
    Architecture:
    1. Node Embedding: Gate Type -> Vector
    2. Local GNN Layers: Encode local connectivity (Circuit Topology)
    3. Global Transformer: Capture long-range dependencies
    4. Fusion: Combine Graph Embedding with Noise Context
    5. Output: Predict Ideal Expectation Value
    """
    def __init__(self, num_gate_types=16, node_dim=64, 
                 num_gnn_layers=2, transformer_heads=4, noise_context_dim=5):
        super().__init__()
        
        # 1. Gate Embeddings
        self.node_embedding = nn.Embedding(num_gate_types, node_dim)
        self.param_encoder = nn.Linear(1, node_dim)
        
        # 2. Graph Transformer Layers
        self.gnn_layers = nn.ModuleList([
            TransformerConv(node_dim, node_dim // transformer_heads, 
                           heads=transformer_heads, concat=True)
            for _ in range(num_gnn_layers)
        ])
        
        # 3. Global Pooling
        self.pool = global_mean_pool
        
        # 4. Noise Context Encoder
        # Encodes: [z0_noisy, zz_noisy, n_qubits, depth, noise_scale]
        self.context_encoder = nn.Linear(noise_context_dim, node_dim)
        
        # 5. Final Regressor
        self.regressor = nn.Sequential(
            nn.Linear(node_dim + node_dim, 64),  # Graph + Context
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output: ⟨Z₀⟩_ideal
        )
```

**What to say:**
> "16 gate types, 64-dimensional embeddings. Two TransformerConv layers for topology. Context encoder for noise information. Final MLP outputs the prediction."

---

## 4. FORWARD PASS (Data Flow)
**File:** `models/qem_former.py` (Lines 76-113)

Shows how data flows through the model.

```python
def forward(self, x, edge_index, batch, global_attr):
    """
    x: [Num_Nodes, 2] - (GateID, Params)
    edge_index: [2, Num_Edges] - Graph connectivity
    batch: [Num_Nodes] - Graph assignment
    global_attr: [Batch_Size, 5] - Noise context
    """
    # A. Node Embedding
    gate_ids = x[:, 0].long()
    params = x[:, 1].unsqueeze(1)
    h = self.node_embedding(gate_ids) + self.param_encoder(params)
    
    # B. Graph Neural Network (Local Topology)
    for gnn in self.gnn_layers:
        h = F.relu(gnn(h, edge_index))
    
    # C. Global Pooling (Aggregate)
    graph_embed = self.pool(h, batch)  # [Batch, 64]
    
    # D. Context Fusion (Inject Noise Info)
    context_embed = F.relu(self.context_encoder(global_attr))
    combined = torch.cat([graph_embed, context_embed], dim=1)  # [Batch, 128]
    
    # E. Prediction
    return self.regressor(combined)  # [Batch, 1]
```

**What to say:**
> "Embed gates, pass through GNN layers, pool to get graph embedding, fuse with noise context, predict ideal value."

---

## 5. TRAINING LOOP
**File:** `train_qem.py` (Lines 90-140)

Standard PyTorch training with LR scheduling.

```python
for epoch in range(EPOCHS):  # 100 epochs
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Reshape global attributes for batch
        g_attr = batch.global_attr.reshape(batch.num_graphs, -1)
        
        # Forward pass
        output = model(batch.x, batch.edge_index, batch.batch, g_attr)
        
        # MSE Loss
        loss = criterion(output.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)  # ReduceLROnPlateau
    
    # Save best model
    if avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), "qem_former.pth")
```

**What to say:**
> "Standard training loop with MSE loss. We use ReduceLROnPlateau - if validation loss plateaus for 10 epochs, we halve the learning rate."

---

## 6. IDEAL VALUE CALCULATION (Statevector)
**File:** `data_gen_advanced.py` (Lines 440-460)

How we compute exact ground truth.

```python
from qiskit.quantum_info import Statevector

# Get exact statevector (no noise)
qc_no_meas = qc.remove_final_measurements(inplace=False)
sv = Statevector.from_instruction(qc_no_meas)
probs = sv.probabilities()

# Calculate ⟨Z₀⟩ from probabilities
z0_ideal = 0
for idx, p in enumerate(probs):
    bit_0 = (idx >> 0) & 1  # Extract qubit 0
    sign = 1 if bit_0 == 0 else -1  # |0⟩ → +1, |1⟩ → -1
    z0_ideal += sign * p

# Calculate ⟨Z₀Z₁⟩ correlation
zz_ideal = 0
for idx, p in enumerate(probs):
    bit_0 = (idx >> 0) & 1
    bit_1 = (idx >> 1) & 1
    sign = (1 if bit_0 == 0 else -1) * (1 if bit_1 == 0 else -1)
    zz_ideal += sign * p
```

**What to say:**
> "Statevector gives us exact probabilities. We sum over all basis states, weighted by +1 or -1 based on qubit values. This is mathematically exact ground truth."

---

## QUICK REFERENCE: Line Numbers

| Concept | File | Lines |
|---------|------|-------|
| Pauli Twirling | `data_gen_advanced.py` | 117-220 |
| Graph Conversion | `data_gen_advanced.py` | 33-114 |
| Data Generation Main | `data_gen_advanced.py` | 267-360 |
| Statevector Ground Truth | `data_gen_advanced.py` | 440-460 |
| QEMFormer Class | `models/qem_former.py` | 8-74 |
| Forward Pass | `models/qem_former.py` | 76-113 |
| Training Loop | `train_qem.py` | 90-140 |
