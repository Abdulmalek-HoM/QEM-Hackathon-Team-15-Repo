
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

# --- Observable Calculation Helpers ---

def calculate_z0_expectation(counts):
    """Calculate <Z_0> from measurement counts."""
    z0 = 0
    total = 0
    for bitstring, count in counts.items():
        # Qiskit bitstring is little-endian (qN...q0). So q0 is the LAST char.
        q0_val = int(bitstring[-1])
        sign = 1 if q0_val == 0 else -1
        z0 += sign * count
        total += count
    return z0 / total if total > 0 else 0

def calculate_zz_correlation(counts, qubit_a=0, qubit_b=1):
    """Calculate <Z_a Z_b> correlation from measurement counts."""
    zz = 0
    total = 0
    for bitstring, count in counts.items():
        # Qiskit bitstring is little-endian (qN...q0)
        # bitstring[-1] is q0, bitstring[-2] is q1, etc.
        try:
            za = int(bitstring[-(qubit_a + 1)])
            zb = int(bitstring[-(qubit_b + 1)])
            # Z eigenvalues: |0> -> +1, |1> -> -1
            sign_a = 1 if za == 0 else -1
            sign_b = 1 if zb == 0 else -1
            zz += sign_a * sign_b * count
            total += count
        except IndexError:
            continue
    return zz / total if total > 0 else 0

def calculate_global_parity(counts):
    """Calculate global parity <Z_0 Z_1 ... Z_n> from measurement counts."""
    parity = 0
    total = 0
    for bitstring, count in counts.items():
        # Count number of 1s - if even parity = +1, if odd parity = -1
        num_ones = bitstring.count('1')
        sign = 1 if num_ones % 2 == 0 else -1
        parity += sign * count
        total += count
    return parity / total if total > 0 else 0

# --- Main Data Generation ---

def generate_advanced_dataset(n_samples=1000, min_qubits=5, max_qubits=20, chunk_id=0, 
                               include_zz=True, include_parity=False, noise_scale=1.5):
    """
    Generates dataset using Clifford circuits + CDR + Pauli Twirling.
    
    Args:
        n_samples: Number of samples to generate
        min_qubits: Minimum qubit count per circuit
        max_qubits: Maximum qubit count per circuit
        chunk_id: Identifier for this data chunk
        include_zz: Include ZZ correlation observable
        include_parity: Include global parity observable
        noise_scale: Noise intensity (higher = more noise)
    
    Saves as .pt file with multi-observable targets.
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        
    builder = QEMGraphBuilder()
    data_list = []
    
    # Simulators
    sim_ideal = AerSimulator(method='stabilizer')  # Fast for Clifford
    noise_model = utils.build_noise_model(scale=noise_scale)
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    print(f"Generating Chunk {chunk_id}: {n_samples} samples...")
    print(f"  Qubits: {min_qubits}-{max_qubits}, Noise Scale: {noise_scale}")
    print(f"  Observables: Z0=True, ZZ={include_zz}, Parity={include_parity}")
    
    iterator = tqdm(range(n_samples), desc=f"Chunk {chunk_id}") if use_tqdm else range(n_samples)
    
    for i in iterator:
        n_q = np.random.randint(min_qubits, max_qubits + 1)
        depth = np.random.randint(n_q, n_q * 3)  # Depth scales with qubits
        
        # 1. Generate Random Clifford Circuit
        qc, _ = utils.create_random_clifford_circuit(n_q, depth)
        qc.measure_all()
        
        # 2. Ideal Execution
        qc_sim = qc.copy()
        counts_ideal = sim_ideal.run(qc_sim, shots=2000).result().get_counts()
        
        # Calculate observables - Ideal
        z0_ideal = calculate_z0_expectation(counts_ideal)
        zz_ideal = calculate_zz_correlation(counts_ideal, 0, 1) if include_zz and n_q >= 2 else 0.0
        parity_ideal = calculate_global_parity(counts_ideal) if include_parity else 0.0
        
        # 3. Noisy Execution with Pauli Twirling
        qc_twirled = apply_pauli_twirling(qc)
        qc_noisy_transpiled = transpile(qc_twirled, sim_noisy)
        
        counts_noisy = sim_noisy.run(qc_noisy_transpiled, shots=2000).result().get_counts()
        
        # Calculate observables - Noisy
        z0_noisy = calculate_z0_expectation(counts_noisy)
        zz_noisy = calculate_zz_correlation(counts_noisy, 0, 1) if include_zz and n_q >= 2 else 0.0
        parity_noisy = calculate_global_parity(counts_noisy) if include_parity else 0.0
        
        # 4. Graph Conversion with extended features
        # Global Features: [z0_noisy, zz_noisy, qubit_count, depth, noise_scale]
        global_feats = [z0_noisy, zz_noisy, float(n_q), float(depth), noise_scale]
        
        graph_data = builder.circuit_to_graph(qc, global_features=global_feats)
        
        # Multi-observable target: [z0_ideal, zz_ideal, parity_ideal]
        graph_data.y = torch.tensor([z0_ideal], dtype=torch.float)  # Primary target
        graph_data.y_z0 = torch.tensor([z0_ideal], dtype=torch.float)
        graph_data.y_zz = torch.tensor([zz_ideal], dtype=torch.float)
        graph_data.y_parity = torch.tensor([parity_ideal], dtype=torch.float)
        
        # Store metadata for analysis
        graph_data.n_qubits = n_q
        graph_data.depth = depth
        
        data_list.append(graph_data)
        
        if not use_tqdm and i % 200 == 0:
            print(f"  Sample {i}: Z0_ideal={z0_ideal:.3f}, ZZ_ideal={zz_ideal:.3f}")
            
    # Save
    save_path = os.path.join(DATASET_DIR, f"train_data_chunk_{chunk_id}.pt")
    torch.save(data_list, save_path)
    print(f"Saved {len(data_list)} graphs to {save_path}")
    
    return data_list

def generate_large_dataset(total_samples=5000, chunk_size=500, **kwargs):
    """
    Generate a large dataset in chunks for memory efficiency.
    
    Args:
        total_samples: Total number of samples to generate
        chunk_size: Samples per chunk file
        **kwargs: Other args passed to generate_advanced_dataset
    """
    n_chunks = (total_samples + chunk_size - 1) // chunk_size
    print(f"Generating {total_samples} samples in {n_chunks} chunks...")
    
    all_data = []
    for chunk_id in range(n_chunks):
        samples_this_chunk = min(chunk_size, total_samples - chunk_id * chunk_size)
        chunk_data = generate_advanced_dataset(
            n_samples=samples_this_chunk,
            chunk_id=chunk_id,
            **kwargs
        )
        all_data.extend(chunk_data)
        
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Total samples: {len(all_data)}")
    return all_data

def generate_mixed_dataset(n_samples=2000, min_qubits=4, max_qubits=8, noise_scale=1.5, chunk_id=100,
                            clifford_frac=0.40, qaoa_frac=0.35, variational_frac=0.25):
    """
    Generate a MIXED dataset with Clifford, QAOA, and Variational circuits.
    Uses STATEVECTOR for accurate ground truth on non-Clifford circuits.
    
    This improves OOD generalization significantly.
    
    Args:
        n_samples: Number of samples to generate
        min_qubits: Minimum qubit count
        max_qubits: Maximum qubit count
        noise_scale: Noise intensity
        chunk_id: Chunk identifier
        clifford_frac: Fraction of Clifford circuits (default 0.40)
        qaoa_frac: Fraction of QAOA circuits (default 0.35)
        variational_frac: Fraction of Variational circuits (default 0.25)
    """
    from qiskit.quantum_info import Statevector
    
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    builder = QEMGraphBuilder()
    data_list = []
    
    noise_model = utils.build_noise_model(scale=noise_scale)
    sim_noisy = AerSimulator(noise_model=noise_model)
    sim_stabilizer = AerSimulator(method='stabilizer')
    
    # Build circuit type distribution from fractions
    clifford_count = int(clifford_frac * 100)
    qaoa_count = int(qaoa_frac * 100)
    variational_count = 100 - clifford_count - qaoa_count
    circuit_types = ['clifford'] * clifford_count + ['qaoa'] * qaoa_count + ['variational'] * variational_count
    
    print(f"Generating MIXED Dataset (Chunk {chunk_id}): {n_samples} samples")
    print(f"  Distribution: {clifford_frac*100:.0f}% Clifford, {qaoa_frac*100:.0f}% QAOA, {variational_frac*100:.0f}% Variational")
    print(f"  Qubits: {min_qubits}-{max_qubits}, Noise Scale: {noise_scale}")
    
    iterator = tqdm(range(n_samples), desc="Mixed Gen") if use_tqdm else range(n_samples)
    
    for i in iterator:
        n_q = np.random.randint(min_qubits, max_qubits + 1)
        circuit_type = np.random.choice(circuit_types)
        
        # Generate circuit based on type
        if circuit_type == 'clifford':
            depth = np.random.randint(n_q, n_q * 3)
            qc, _ = utils.create_random_clifford_circuit(n_q, depth)
            use_stabilizer = True
        elif circuit_type == 'qaoa':
            p = np.random.randint(1, 4)
            qc, _ = utils.create_qaoa_circuit(n_q, p=p)
            depth = p * 3 * n_q  # Approximate depth
            use_stabilizer = False
        else:  # variational
            depth = np.random.randint(3, 8)
            qc, _ = utils.create_variational_circuit(n_q, depth)
            use_stabilizer = False
        
        qc.measure_all()
        
        # IDEAL: Use statevector for ALL circuits (most accurate)
        try:
            qc_no_meas = qc.remove_final_measurements(inplace=False)
            sv = Statevector.from_instruction(qc_no_meas)
            probs = sv.probabilities()
            
            # Calculate <Z_0>
            z0_ideal = 0
            for idx, p in enumerate(probs):
                bit_0 = (idx >> 0) & 1
                sign = 1 if bit_0 == 0 else -1
                z0_ideal += sign * p
                
            # Calculate <Z_0 Z_1>
            zz_ideal = 0
            if n_q >= 2:
                for idx, p in enumerate(probs):
                    bit_0 = (idx >> 0) & 1
                    bit_1 = (idx >> 1) & 1
                    sign = (1 if bit_0 == 0 else -1) * (1 if bit_1 == 0 else -1)
                    zz_ideal += sign * p
        except Exception as e:
            # Fallback to stabilizer for Clifford
            if use_stabilizer:
                counts = sim_stabilizer.run(qc.copy(), shots=2000).result().get_counts()
                z0_ideal = calculate_z0_expectation(counts)
                zz_ideal = calculate_zz_correlation(counts, 0, 1) if n_q >= 2 else 0.0
            else:
                continue  # Skip failed circuits
        
        # NOISY execution
        qc_twirled = apply_pauli_twirling(qc)
        qc_noisy = transpile(qc_twirled, sim_noisy)
        counts_noisy = sim_noisy.run(qc_noisy, shots=2000).result().get_counts()
        
        z0_noisy = calculate_z0_expectation(counts_noisy)
        zz_noisy = calculate_zz_correlation(counts_noisy, 0, 1) if n_q >= 2 else 0.0
        
        # Build graph
        global_feats = [z0_noisy, zz_noisy, float(n_q), float(depth), noise_scale]
        graph_data = builder.circuit_to_graph(qc, global_features=global_feats)
        
        graph_data.y = torch.tensor([z0_ideal], dtype=torch.float)
        graph_data.y_z0 = torch.tensor([z0_ideal], dtype=torch.float)
        graph_data.y_zz = torch.tensor([zz_ideal], dtype=torch.float)
        graph_data.y_parity = torch.tensor([0.0], dtype=torch.float)  # Placeholder for compatibility
        graph_data.n_qubits = n_q
        graph_data.depth = depth
        # Note: circuit_type stored as metadata only, not as graph attribute to avoid collation issues
        
        data_list.append(graph_data)
    
    # Save
    save_path = os.path.join(DATASET_DIR, f"train_data_mixed_{chunk_id}.pt")
    torch.save(data_list, save_path)
    print(f"Saved {len(data_list)} mixed graphs to {save_path}")
    
    return data_list

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate QEM training dataset")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--min-qubits", type=int, default=4, help="Min qubits")
    parser.add_argument("--max-qubits", type=int, default=10, help="Max qubits")
    parser.add_argument("--chunk-id", type=int, default=0, help="Chunk ID")
    parser.add_argument("--large", action="store_true", help="Generate large dataset (5000 samples)")
    parser.add_argument("--mixed", action="store_true", help="Generate mixed circuit dataset for OOD")
    parser.add_argument("--noise-scale", type=float, default=1.5, help="Noise scale factor")
    parser.add_argument("--qaoa-frac", type=float, default=0.35, help="Fraction of QAOA circuits (default 0.35)")
    parser.add_argument("--clifford-frac", type=float, default=0.40, help="Fraction of Clifford circuits (default 0.40)")
    parser.add_argument("--variational-frac", type=float, default=0.25, help="Fraction of Variational circuits (default 0.25)")
    
    args = parser.parse_args()
    
    if args.mixed:
        generate_mixed_dataset(
            n_samples=args.samples,
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
            noise_scale=args.noise_scale,
            chunk_id=args.chunk_id,
            clifford_frac=args.clifford_frac,
            qaoa_frac=args.qaoa_frac,
            variational_frac=args.variational_frac
        )
    elif args.large:
        generate_large_dataset(
            total_samples=5000,
            chunk_size=500,
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
            noise_scale=args.noise_scale
        )
    else:
        generate_advanced_dataset(
            n_samples=args.samples,
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
            chunk_id=args.chunk_id,
            noise_scale=args.noise_scale
        )


