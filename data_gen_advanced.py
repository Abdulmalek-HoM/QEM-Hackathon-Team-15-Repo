
import os
import torch
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.converters import circuit_to_dag
from torch_geometric.data import Data
import mitiq
from mitiq import cdr

import utils  # Import existing utils for consistency if needed, but we'll build fresh advanced logic

# --- Configuration ---
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

class QEMGraphBuilder:
    """
    Converts Qiskit Circuits to PyTorch Geometric Graphs.
    Nodes: Quantum Gates
    Edges: Wires (Qubit dependecies)
    """
    def __init__(self):
        # Vocabulary for gate types
        self.gate_vocab = {
            "h": 0, "s": 1, "sdg": 2, "x": 3, "y": 4, "z": 5, "id": 6, 
            "cx": 7, "cz": 8, "swap": 9, "rx": 10, "ry": 11, "rz": 12, 
            "measure": 13, "barrier": 14
        }
    
    def circuit_to_graph(self, qc: QuantumCircuit, global_features=None) -> Data:
        """
        Converts a quantum circuit to a Graph Data object.
        """
        dag = circuit_to_dag(qc)
        node_features = []
        edge_index = [[], []] # [source, target]
        
        # We need to traverse the DAG to get nodes in execution order if possible, 
        # or topological sort. Qiskit DAG is already topological.
        
        # Map DAGNodes to integers
        node_map = {}
        msg_idx = 0
        
        # Iterate over topological op nodes
        for node in dag.topological_op_nodes():
            # 1. Feature Extraction
            gate_name = node.name
            gate_id = self.gate_vocab.get(gate_name, 15) # 15 = Unknown
            
            # Simple embedding: [GateID, Parameter(if any)]
            # For now just GateID. In future, add params (angles).
            # We can also add qubit indices as node features or edge features.
            
            # Let's use One-Hot or Embedding index for GateID
            feat = [gate_id]
            
            # Add rotation params if available
            if hasattr(node.op, 'params') and len(node.op.params) > 0:
                # Take first param if exists, else 0. Normalize?
                p = float(node.op.params[0]) if isinstance(node.op.params[0], (int, float)) else 0.0
                feat.append(p)
            else:
                feat.append(0.0)
                
            node_features.append(feat)
            node_map[node] = msg_idx
            msg_idx += 1
            
        # 2. Build Edges (Connectivity)
        # We iterate over qubits (wires) and connect sequential ops
        for wire in qc.qubits:
            # Get nodes on this wire
            # dag.nodes_on_wire(wire) returns iterator of nodes
            # We filter for op_nodes only usually, but let's see.
            
            # Qiskit DAG structure is a bit complex. 
            # Simplified approach: Iterate topological nodes and track "previous node" per qubit.
            pass
        
        # Alternative Edge Building: 
        # Iterate over all op nodes. For each qarg in node, find the previous op that acted on it.
        # This is strictly DAG based.
        
        last_node_on_qubit = {q: -1 for q in qc.qubits} # Stores index of last node
        
        # Re-iterate to build edges based on the order we defined above (node_features)
        current_idx = 0
        for node in dag.topological_op_nodes():
            for q in node.qargs:
                prev_idx = last_node_on_qubit[q]
                if prev_idx != -1:
                    # Edge from Prev -> Current
                    edge_index[0].append(prev_idx)
                    edge_index[1].append(current_idx)
                    
                    # We can add edge attributes (e.g., which qubit is this wire?)
                    # edge_attr.append(qubit_index)
                
                last_node_on_qubit[q] = current_idx
            current_idx += 1
            
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        
        if global_features is not None:
             data.global_attr = torch.tensor(global_features, dtype=torch.float)
             
        return data

# --- Pauli Twirling ---
def apply_pauli_twirling(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Applies Pauli Twirling to CNOT gates in the circuit.
    Converts coherent errors into stochastic Pauli errors.
    """
    twirled_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    
    for instruction in qc.data:
        op = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        
        if op.name == 'cx': # Target only CNOTs for now (dominating error source)
            # 1. Pick random Pauli pair (P_i, P_j) such that (P_i \otimes P_j) CNOT = CNOT (P_k \otimes P_l)
            # Actually, standard twirling:
            # Apply random Paulis BEFORE and match them AFTER to cancel out logically.
            # Setup: -- P_c -- * -- P_c' --
            #        -- P_t -- X -- P_t' --
            # Where P' are chosen such that the total operation is still CNOT.
            
            # Valid pairs for CNOT twirling (from literature, e.g. Wallman/Emerson)
            # 16 sets of Pauli gates.
            # Simple subset: {I, X, Y, Z}
            
            # We can use qiskit.quantum_info.pauli_basis or simple hardcoded logic for CNOT
            # CNOT commutes with Z_c, X_t. Anticommutes mixed.
            # Detailed implementation:
            # QC_Twirled = P_before * CX * P_after
            
            # Pick random Pauli for Control and Target BEFORE
            # P_c, P_t
            # Calculate required P_c', P_t' AFTER to restore Identity
            # CNOT (P_c \otimes P_t) CNOT = (P_c' \otimes P_t')
            
            # Easier way: Use Mitiq's twirling or implement simple random implementation.
            # Let's implement manually for control.
            
            # Random Pauli indices: 0=I, 1=X, 2=Y, 3=Z
            # But in gate terms: I, X, Y, Z
            paulis = ['id', 'x', 'y', 'z']
            
            # Random selection
            idx_c = np.random.randint(4)
            idx_t = np.random.randint(4)
            
            pc_gate = paulis[idx_c]
            pt_gate = paulis[idx_t]
            
            # Logic to find correction gates:
            # We want: (Pc' x Pt') . CX . (Pc x Pt) == CX
            # So: (Pc' x Pt') == CX . (Pc x Pt)^dag . CX^dag
            # Since Paulis are self-inverse (mostly, up to phase), and we ignore global phase:
            # We just need to push (Pc x Pt) through CNOT.
            
            # Rules:
            # I x I -> I x I
            # I x X -> I x X
            # I x Y -> Z x Y (Phase?) -> Z x Y
            # I x Z -> Z x Z
            # X x I -> X x X
            # X x X -> X x I
            # X x Y -> Y x Z
            # X x Z -> Y x Y
            # ... and so on.
            
            # Let's use specific known set of twirls to avoid bugs.
            # Or just use `mitiq.cdt.utils.pauli_twirling` if available? 
            # Mitiq has `mitiq.dd.twirling`. Let's assume we build a simple efficient one.
            
            # We will insert the PRE gates, the CNOT, and the POST gates.
            
            # PRE Gates
            if pc_gate != 'id': getattr(twirled_qc, pc_gate)(qubits[0])
            if pt_gate != 'id': getattr(twirled_qc, pt_gate)(qubits[1])
            
            # CNOT
            twirled_qc.cx(qubits[0], qubits[1])
            
            # POST Gates (calculated)
            # Tuple key: (Control_Pauli, Target_Pauli) -> (Control_Post, Target_Post)
            # 0=I, 1=X, 2=Y, 3=Z
            lookup = {
                # Control is I
                (0,0):(0,0), (0,1):(0,1), (0,2):(3,2), (0,3):(3,3),
                # Control is X
                (1,0):(1,1), (1,1):(1,0), (1,2):(2,3), (1,3):(2,2),
                # Control is Y
                (2,0):(2,1), (2,1):(2,0), (2,2):(1,3), (2,3):(1,2),
                # Control is Z
                (3,0):(3,0), (3,1):(3,1), (3,2):(0,2), (3,3):(0,3)
            }
            
            res = lookup[(idx_c, idx_t)]
            res_c_name = paulis[res[0]]
            res_t_name = paulis[res[1]]
            
            if res_c_name != 'id': getattr(twirled_qc, res_c_name)(qubits[0])
            if res_t_name != 'id': getattr(twirled_qc, res_t_name)(qubits[1])

        else:
            # Copy other instructions
            twirled_qc.append(op, qubits, clbits)
            
    return twirled_qc

# --- Main Data Generation ---

def generate_advanced_dataset(n_samples=1000, min_qubits=5, max_qubits=20, chunk_id=0):
    """
    Generates dataset using Clifford circuits + CDR + Pauli Twirling.
    Saves as .pt file.
    """
    builder = QEMGraphBuilder()
    data_list = []
    
    # Simulators
    sim_ideal = AerSimulator(method='stabilizer') # Fast for Clifford
    noise_model = utils.build_noise_model(scale=1.5) # Stronger noise for training
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    print(f"Generating Chunk {chunk_id}: {n_samples} samples...")
    
    for i in range(n_samples):
        n_q = np.random.randint(min_qubits, max_qubits + 1)
        depth = np.random.randint(n_q, n_q*3) # Depth scales with qubits
        
        # 1. Generate Random Clifford Circuit
        qc, _ = utils.create_random_clifford_circuit(n_q, depth)
        qc.measure_all()
        
        # 2. Get GROUND TRUTH (Ideal Expectation of Z on qubit 0 or global parity?)
        # Let's target Global Parity <ZZ...Z> for simplicity or just <Z_0>.
        # Let's do <Z_0> to keep it simple for now, or average magnetization.
        # Ideally QEM is for specific observables. Let's pick Observable O = Z on qubit 0.
        
        qc_sim = qc.copy()
        # Ideal run
        result_ideal = sim_ideal.run(qc_sim, shots=None).result() # Exact probabilities from stabilizer if supported, or shots
        # Aer stabilizer gives counts.
        counts_ideal = sim_ideal.run(qc_sim, shots=1000).result().get_counts()
        
        # Calculate Expectation <Z_0>
        # P(0) - P(1) for qubit 0.
        z0_ideal = 0
        total_shots = 0
        for bitstring, count in counts_ideal.items():
            # Qiskit bitstring is little-endian (qN...q0). So q0 is the LAST char.
            q0_val = int(bitstring[-1]) # 0 or 1
            sign = 1 if q0_val == 0 else -1
            z0_ideal += sign * count
            total_shots += count
        z0_ideal /= total_shots
        
        # 3. Noisy Execution with Pauli Twirling
        # We run the circuit MULTIPLE times with different twirls to average stochastic noise?
        # CDR usually runs the exact circuit (or twirled variants). 
        # Feature for AI: "Noisy Expectation"
        
        # Apply Pauli Twirling ONCE per sample? Or average over duplicates?
        # For training efficiently, let's apply one random twirl per sample 
        # and let the batching handle the averaging effectively, 
        # OR run say 10 twirled shots.
        
        # Let's do: Create Twirled Circuit -> Run on Noisy Backend
        qc_twirled = apply_pauli_twirling(qc)
        qc_noisy_transpiled = transpile(qc_twirled, sim_noisy)
        
        result_noisy = sim_noisy.run(qc_noisy_transpiled, shots=1000).result()
        counts_noisy = result_noisy.get_counts()
        
        # Calculate <Z_0> Noisy
        z0_noisy = 0
        total_shots_n = 0
        for bitstring, count in counts_noisy.items():
            q0_val = int(bitstring[-1])
            sign = 1 if q0_val == 0 else -1
            z0_noisy += sign * count
            total_shots_n += count
        z0_noisy /= total_shots_n
        
        # 4. Graph Conversion
        # Features: Noisy Value usually is an INPUT feature to the model? 
        # Or does the model predict Ideal from Noisy?
        # QEM-Former Input: Circuit Graph.
        # We need to inject "Noisy value" somewhere.
        # Option A: Global Attribute of the graph = [Noisy_Expectation, T1, T2...]
        # Option B: The model just learns to predict scale factors?
        # CDR approach: Train model to predict Ideal from (Noisy, Circuit).
        
        # Let's add Noisy Expectation to "Global Features"
        # Features: [Noisy_Exp, Qubit_Count, Depth]
        global_feats = [z0_noisy, float(n_q), float(depth)]
        
        graph_data = builder.circuit_to_graph(qc, global_features=global_feats)
        graph_data.y = torch.tensor([z0_ideal], dtype=torch.float) # Target
        
        data_list.append(graph_data)
        
        if i % 100 == 0:
            print(f"  Sample {i}: Ideal={z0_ideal:.3f}, Noisy={z0_noisy:.3f}")
            
    # Save
    save_path = os.path.join(DATASET_DIR, f"train_data_chunk_{chunk_id}.pt")
    torch.save(data_list, save_path)
    print(f"Saved {len(data_list)} graphs to {save_path}")

if __name__ == "__main__":
    generate_advanced_dataset(n_samples=500, min_qubits=4, max_qubits=10, chunk_id=0)
