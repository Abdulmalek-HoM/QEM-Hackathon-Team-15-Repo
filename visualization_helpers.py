import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from qiskit.converters import circuit_to_dag
import utils

def plot_error_by_qubit(n_qubits, noise_scale):
    """Simulate per-qubit error distribution."""
    # Simulate realistic per-qubit errors based on noise scale
    np.random.seed(42)
    base_errors = np.random.uniform(0.02, 0.08, n_qubits) * noise_scale
    mitigated_errors = base_errors * np.random.uniform(0.3, 0.7, n_qubits)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_qubits)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_errors, width, label='Noisy', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x + width/2, mitigated_errors, width, label='Mitigated', color='#3498DB', alpha=0.8)
    
    ax.set_xlabel('Qubit Index')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Distribution by Qubit')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{i}' for i in range(n_qubits)])
    ax.legend()
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    return fig

def plot_prediction_scatter(pipeline, n_samples=20):
    """Generate scatter plot of predictions vs actual."""
    predictions = []
    actuals = []
    
    for i in range(n_samples):
        np.random.seed(i)
        n_qubits = np.random.randint(3, 7)
        depth = np.random.randint(5, 20)
        qc, instructions = utils.create_random_clifford_circuit(n_qubits, depth)
        qc.measure_all()
        
        try:
            pred, _, _ = pipeline.predict(qc, instructions)
            true_val, _ = pipeline.get_ground_truth(qc)
            predictions.append(pred)
            actuals.append(true_val)
        except:
            continue
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actuals, predictions, alpha=0.7, c='#3498DB', s=80, edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.7, label='Perfect')
    
    ax.set_xlabel('Actual (Ideal)')
    ax.set_ylabel('Predicted (CDRFormer)')
    ax.set_title('Prediction vs Actual')
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    return fig

def plot_connectivity_heatmap(qc):
    """Generate qubit connectivity heatmap from circuit."""
    n_qubits = qc.num_qubits
    connectivity = np.zeros((n_qubits, n_qubits))
    
    for instr in qc.data:
        if len(instr.qubits) == 2:
            q1 = qc.find_bit(instr.qubits[0]).index
            q2 = qc.find_bit(instr.qubits[1]).index
            connectivity[q1, q2] += 1
            connectivity[q2, q1] += 1
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(connectivity, cmap='Blues')
    
    ax.set_xticks(np.arange(n_qubits))
    ax.set_yticks(np.arange(n_qubits))
    ax.set_xticklabels([f'Q{i}' for i in range(n_qubits)])
    ax.set_yticklabels([f'Q{i}' for i in range(n_qubits)])
    ax.set_title('Qubit Connectivity & Error Rate')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('2-Qubit Gate Count', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    return fig

def plot_circuit_dag(qc):
    """Visualize the circuit as a Directed Acyclic Graph (DAG)."""
    dag = circuit_to_dag(qc)
    
    # Build networkx graph from DAG
    G = nx.DiGraph()
    
    # Gate type colors
    gate_colors = {
        'h': '#3498DB',      # Blue - Hadamard
        's': '#9B59B6',      # Purple - S gate
        'sdg': '#9B59B6',    # Purple - S dagger
        'x': '#E74C3C',      # Red - Pauli X
        'y': '#F39C12',      # Orange - Pauli Y
        'z': '#2ECC71',      # Green - Pauli Z
        'cx': '#1ABC9C',     # Teal - CNOT
        'cz': '#16A085',     # Dark teal - CZ
        'rx': '#E91E63',     # Pink - RX
        'ry': '#FF5722',     # Deep orange - RY
        'rz': '#00BCD4',     # Cyan - RZ
        'measure': '#95A5A6', # Gray - Measurement
        'barrier': '#34495E', # Dark gray - Barrier
    }
    
    node_labels = {}
    node_colors = []
    node_map = {}
    idx = 0
    
    # Add nodes (gates)
    for node in dag.topological_op_nodes():
        node_id = f"{node.name}_{idx}"
        node_map[node] = node_id
        G.add_node(node_id)
        
        # Create label with qubit info
        qubits = [qc.find_bit(q).index for q in node.qargs]
        if len(qubits) == 1:
            node_labels[node_id] = f"{node.name.upper()}\nq{qubits[0]}"
        else:
            node_labels[node_id] = f"{node.name.upper()}\nq{qubits[0]},q{qubits[1]}"
        
        node_colors.append(gate_colors.get(node.name, '#7F8C8D'))
        idx += 1
    
    # Add edges (dependencies)
    last_node_on_qubit = {}
    idx = 0
    for node in dag.topological_op_nodes():
        node_id = node_map[node]
        for q in node.qargs:
            qubit_idx = qc.find_bit(q).index
            if qubit_idx in last_node_on_qubit:
                G.add_edge(last_node_on_qubit[qubit_idx], node_id)
            last_node_on_qubit[qubit_idx] = node_id
        idx += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(G.nodes()) > 0:
        # Use layered layout for DAG
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback to spring layout if graphviz not available
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#555555', 
                               arrows=True, arrowsize=15, 
                               connectionstyle='arc3,rad=0.1',
                               alpha=0.7)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=1500, alpha=0.9)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, 
                                font_size=8, font_color='white',
                                font_weight='bold')
    
    ax.set_title('Circuit as Directed Acyclic Graph (DAG)', fontsize=14, color='white')
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.axis('off')
    
    return fig, len(G.nodes()), len(G.edges())
