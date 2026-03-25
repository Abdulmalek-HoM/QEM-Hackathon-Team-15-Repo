import streamlit as st
import graphviz

def render_codebase_tab():
    st.header("🗂️ Codebase Architecture & Files Overview")
    st.info("A comprehensive guide to all Python files in the repository, explaining their purpose in the QEM pipeline.")
    
    st.markdown("### 1. File Dependency Map")
    
    # Create Graphviz diagram representing codebase architecture
    graph = graphviz.Digraph(format='svg')
    graph.attr(rankdir='LR', size='10,6')
    
    # Define Nodes with colors
    with graph.subgraph(name='cluster_data') as c:
        c.attr(style='filled', color='#EAFAF1')
        c.node_attr.update({'style': 'filled', 'color': 'white'})
        c.node('data_gen', 'data_gen_advanced.py', fillcolor='#2ECC71', fontcolor='white')
        c.node('utils', 'utils.py', fillcolor='#27AE60', fontcolor='white')
        c.node('schemas', 'backend/schemas.py', fillcolor='#1E8449', fontcolor='white')
        c.attr(label='🧬 Data Generation Layer')

    with graph.subgraph(name='cluster_ai') as c:
        c.attr(style='filled', color='#F5EEF8')
        c.node_attr.update({'style': 'filled', 'color': 'white'})
        c.node('cdr_model', 'models/cdr_former.py', fillcolor='#9B59B6', fontcolor='white')
        c.node('train', 'train_qem.py', fillcolor='#8E44AD', fontcolor='white')
        c.attr(label='🧠 Artificial Intelligence Layer')

    with graph.subgraph(name='cluster_eval') as c:
        c.attr(style='filled', color='#EBF5FB')
        c.node_attr.update({'style': 'filled', 'color': 'white'})
        c.node('pipeline', 'backend/pipeline.py', fillcolor='#3498DB', fontcolor='white')
        c.node('benchmark', 'benchmark_suite.py', fillcolor='#2980B9', fontcolor='white')
        c.node('scripts', 'scripts/*.py\n(Ablation Studies)', fillcolor='#21618C', fontcolor='white')
        c.attr(label='🧪 Evaluation Layer')

    with graph.subgraph(name='cluster_ui') as c:
        c.attr(style='filled', color='#FDEDEC')
        c.node_attr.update({'style': 'filled', 'color': 'white'})
        c.node('dashboard', 'dashboard.py', fillcolor='#E74C3C', fontcolor='white')
        c.node('tabs', 'dashboard_tabs/*.py', fillcolor='#C0392B', fontcolor='white')
        c.attr(label='🖥️ Dashboard UI')

    # Dependencies
    graph.edge('utils', 'data_gen')
    graph.edge('data_gen', 'schemas')
    graph.edge('schemas', 'train')
    graph.edge('cdr_model', 'train')
    graph.edge('train', 'pipeline')
    graph.edge('cdr_model', 'pipeline')
    graph.edge('pipeline', 'benchmark')
    graph.edge('pipeline', 'scripts')
    graph.edge('benchmark', 'dashboard')
    graph.edge('scripts', 'dashboard')
    graph.edge('pipeline', 'dashboard')
    graph.edge('dashboard', 'tabs')

    st.graphviz_chart(graph)
    
    st.markdown("---")
    st.markdown("### 2. File Explanations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🧬 Data Generation Layer", expanded=True):
            st.markdown("""
            **`utils.py`**
            *   **Purpose:** The foundation. Contains helper functions to create quantum circuits via Qiskit (`create_random_clifford_circuit`, `create_qaoa_circuit`, `create_variational_circuit`).
            *   **Key Concept:** Also builds the reproducible simulated physics noise models (`build_noise_model`) containing $T_1$, $T_2$, and readout error profiles.
            
            **`data_gen_advanced.py`**
            *   **Purpose:** Orchestrates the CDR data collection. Generates massive sets of $(x_{noisy}, y_{ideal})$ pairs.
            *   **Key Concept:** Implements **Pauli Twirling** on the generated circuits to Markovianize the noisy targets, acting as a crucial pre-processing step for the Neural Network.
            
            **`backend/schemas.py`**
            *   **Purpose:** Defines the `QEMGraphBuilder` class.
            *   **Key Concept:** Automatically converts Qiskit `QuantumCircuit` DAGs into PyTorch Geometric `Data` objects, translating physics constructs into AI arrays. Builds node embeddings based on quantum gate types.
            """)
            
        with st.expander("🧠 Artificial Intelligence Layer", expanded=True):
            st.markdown("""
            **`models/cdr_former.py`**
            *   **Purpose:** The core neural network architecture.
            *   **Key Concept:** Defines the mathematically-backed Graph Attention Network using `TransformerConv` layers from PyTorch Geometric, coupled to a Multi-Layer Perceptron (MLP) head to output continuous expectation values.
            
            **`train_qem.py`**
            *   **Purpose:** The standard model training loop.
            *   **Key Concept:** Takes the `.pt` datasets from `data_gen_advanced.py`, sets up the `Adam` optimizer and MSE loss functions, loops through epochs, and saves the final converged weights to `weights/cdr_former.pth`.
            """)

    with col2:
        with st.expander("🧪 Evaluation Layer", expanded=True):
            st.markdown("""
            **`backend/pipeline.py`**
            *   **Purpose:** The inference wrapper. Designed to act like a production API.
            *   **Key Concept:** You pass an arbitrary `QuantumCircuit` to `pipeline.predict()`, and it seamlessly converts it to an AI graph, runs the AI model, and returns the mitigated output. Also provides Zero Noise Extrapolation (ZNE) physics baselines via Mitiq.
            
            **`benchmark_suite.py`**
            *   **Purpose:** Evaluates the `pipeline.py` outputs statistically over thousands of unseen validation circuits.
            *   **Key Concept:** Generates `assets/benchmark_results.json` mapping out the basic in-distribution (Clifford) and out-of-distribution (QAOA/Variational) error rates.
            
            **`scripts/*.py`** (Ablation Studies)
            *   Contains experimental python files testing specific hypotheses, like testing scalability (`scalability_benchmark.py`), noise-robutsness (`noise_profile_benchmark.py`), or verifying that our QAOA win rate failure was solved by training composition (`data_composition_ablation.py`).
            """)
            
        with st.expander("🖥️ Dashboard UI", expanded=True):
            st.markdown("""
            **`dashboard.py`**
            *   **Purpose:** The entry point for this Streamlit web application.
            *   **Key Concept:** Orchestrates layout, applies CSS, initializes the ML `pipeline`, and serves as the navigation host.
            
            **`dashboard_tabs/*.py`**
            *   **Purpose:** Modular components containing exactly what is rendering right now (like `tab_codebase.py`).
            *   **Key Concept:** Keeps the massive UI frontend code cleanly separated from backend execution logic.
            """)
