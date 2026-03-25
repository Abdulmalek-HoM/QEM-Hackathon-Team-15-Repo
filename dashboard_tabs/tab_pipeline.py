import streamlit as st
import graphviz
import pandas as pd

def render_pipeline_tab():
    st.header("⚙️ Pipeline Architecture & Model Sub-Systems")
    st.info("Comprehensive breakdown of the AI components, data flow, and training hyperparameters.")
    
    st.subheader("1. System Block Diagram")
    
    graph = graphviz.Digraph(format='svg')
    graph.attr(rankdir='LR', size='10,4')
    
    with graph.subgraph(name='cluster_0') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update({'style': 'filled', 'color': 'white', 'fontcolor': 'black'})
        c.node('qasm', 'Circuit File .qasm')
        c.node('parser', 'Qiskit Parser')
        c.edge('qasm', 'parser')
        c.node('dag', 'DAG Graph\n(Nodes, Edges)')
        c.edge('parser', 'dag')
        c.node('global', 'Global Features')
        c.edge('parser', 'global')
        c.attr(label='📁 Data Flow (schemas.py)')

    with graph.subgraph(name='cluster_1') as c:
        c.attr(fillcolor='#E8DAEF', style='filled', color='black')
        c.node('embed', 'Node Embedding\n(64-dim)', fillcolor='#9B59B6', fontcolor='black')
        c.node('conv1', 'TransformerConv 1', fillcolor='#8E44AD', fontcolor='black')
        c.node('conv2', 'TransformerConv 2', fillcolor='#8E44AD', fontcolor='black')
        c.node('pool', 'Global Mean Pool', fillcolor='#9B59B6', fontcolor='black')
        c.edge('embed', 'conv1')
        c.edge('conv1', 'conv2')
        c.edge('conv2', 'pool')
        c.attr(label='🧠 Graph Network (cdr_former.py)')
        
    with graph.subgraph(name='cluster_2') as c:
        c.attr(fillcolor='#D5F5E3', style='filled', color='black')
        c.node('mlp', 'MLP Regressor\n(Hidden=32)', fillcolor='#27AE60', fontcolor='black')
        c.node('out', 'Prediction ⟨Z₀⟩', fillcolor='#1E8449', fontcolor='black')
        c.edge('mlp', 'out')
        c.attr(label='🎯 Prediction Head')

    graph.edge('dag', 'embed')
    graph.edge('pool', 'mlp', label=' concat')
    graph.edge('global', 'mlp', label=' concat')
    
    st.graphviz_chart(graph)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. AI Model Details & Interactive Configizer")
        
        model_selection = st.selectbox("Select Model Component to Inspect", ["Graph Convolution (TransformerConv)", "Node Embedding layer", "MLP Regressor", "Global Pooling"])
        
        if model_selection == "Graph Convolution (TransformerConv)":
            st.latex(r"h_i^{(l+1)} = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{i,j} \Theta h_j^{(l)}")
            st.markdown("""
            **The Engine of Entanglement Representation:**
            This is the core non-linear message-passing protocol. In quantum mechanics, multi-qubit gates (like CNOTs) spread entanglement and noise simultaneously. The `TransformerConv` essentially simulates this physical propagation of crosstalk and correlated errors across the qubit topology. 
            
            By employing **Multi-Head Attention** ($\alpha_{i,j}$), the network learns to dynamically "weigh" which gates corrupted their neighbors the most. Rather than assuming all adjacent gates contribute equally to noise (like standard Convolutional Networks), the attention heads learn the specific signature of *Coherent Error leakage* across control-target qubit pairs over $l$-depth layers.
            """)
        elif model_selection == "Node Embedding layer":
            st.markdown("""
            **Translating Quantum Physics to Feature Space:**
            To feed a Qiskit circuit into a neural net, we must break it down into $V$ nodes (gates) and $E$ edges (wires). The `nn.Embedding` matrix acts as the Rosetta Stone. 
            
            Each of the 20 distinct QASM instructions (e.g., $R_X, R_Z, CX, H$) is mapped from a discrete integer ID into a continuous, dense 64-dimensional vector space. Deep within this latent manifold, the AI discovers geometric analogies—for instance, noting that $X$, $Y$, and $Z$ gates share rotational symmetries, or that Clifford gates behave distinctly from parameterized variations. This embedding provides the crucial context required for the Graph Network to parse *what* operations are accumulating phase errors. 
            """)
        elif model_selection == "MLP Regressor":
            st.markdown("""
            **The Synthesis Matrix (Observer Projection):**
            Once the entire Directed Acyclic Graph of the circuit is encoded into a single 64-D state vector, the Multilayer Perceptron acts as our classical observer measuring the final state. 
            
            Crucially, it is not merely given the graph. It is given a concatenated **69-Dimensional Context Vector** containing `[Graph_Topology, Noisy_Measurement(<Z>), Z-Z_Correlations, Qubit_Count, Depth, Noise_Scale]`. The 3-layer architecture ($32 \rightarrow 16 \rightarrow 1$) acts as a non-linear regression manifold that maps this corrupted physical expectation back into the ideal, zero-noise Hamiltonian expectation surface.
            """)
        elif model_selection == "Global Pooling":
            st.markdown("""
            **The Macroscopic Circuit Collapse (`global_mean_pool`):**
            How do you compress a circuit of 2 qubits and 10 gates into the same tensor size as a circuit of 50 qubits and 1000 gates? 
            
            Standard deep learning might use `max_pooling` to find the most "dominant" feature. But in quantum circuits, errors are *cumulative* and *depolarizing*. By taking the **Mean Vector** across all $N$ gates in the circuit, the pooling layer computes a holistic integral of the entire system's noise signature. This transforms the sequence of localized gate errors into one dense "Circuit Identity Vector" that encapsulates the total error budget of the execution.
            """)
            
    with col2:
        st.subheader("3. Training Configuration")
        
        epochs = st.number_input("Recorded Epochs", value=100, disabled=True)
        batch = st.number_input("Batch Size", value=32, disabled=True)
        lr = st.number_input("Initial Learning Rate", value=0.001, disabled=True)
        
        st.markdown("**Optimizer Tracking:**")
        df_opt = pd.DataFrame({
            "Optimizer": ["Adam"],
            "Loss Function": ["Mean Squared Error (MSE)"],
            "Scheduler": ["ReduceLROnPlateau (patience=10)"],
            "Best Val Loss": ["0.000378 (Cross-Arch) / 0.0094 (Base)"]
        })
        st.dataframe(df_opt, hide_index=True)
        st.caption("Training logic resides in `train_qem.py`.")
