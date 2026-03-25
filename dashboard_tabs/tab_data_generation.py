import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from utils import create_random_clifford_circuit, create_qaoa_circuit, create_variational_circuit, LiveHardwareProfiler

def render_data_generation_tab(pipeline):
    st.header("🧬 Data Generation Pipeline")
    st.info("Interactive control over quantum circuit generation and noise injection for training data.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Mathematical Block Diagram")
        
        # Graphviz block diagram for data gen
        graph = graphviz.Digraph(format='svg')
        graph.attr(rankdir='TB', size='6,6')
        
        graph.node('A', 'Circuit Specifications\n(Qubits, Depth, Type)', shape='parallelogram', style='filled', fillcolor='#D2B4DE')
        graph.node('B', 'Circuit Generator\n(Qiskit Circuit)', shape='box', style='filled', fillcolor='#AED6F1')
        graph.node('C', 'Pauli Twirling\n(Markovianize coherent errors)', shape='box', style='filled', fillcolor='#F9E79F')
        graph.node('D', 'Ideal Setup\n(Statevector Sim)', shape='cylinder', style='filled', fillcolor='#2ECC71', fontcolor='white')
        graph.node('E', 'Noisy Setup\n(AerSim + T1/T2 + Readout)', shape='cylinder', style='filled', fillcolor='#E74C3C', fontcolor='white')
        graph.node('F', 'Data Pair\n(y_ideal, x_noisy)', shape='note', style='filled', fillcolor='#F1C40F')
        
        graph.edge('A', 'B')
        graph.edge('B', 'C', label=' for CNOTs')
        graph.edge('B', 'D', label=' Exact Math')
        graph.edge('C', 'E', label=' Stochastic Noise')
        graph.edge('D', 'F')
        graph.edge('E', 'F')
        
        st.graphviz_chart(graph)
        
    with col2:
        st.subheader("2. Interactive Generation Control")
        circuit_type = st.selectbox(
            "Select Training Circuit Type",
            ["Random Clifford", "QAOA (MaxCut)", "Variational Ansatz"]
        )
        
        n_qubits = st.slider("Number of Qubits", 2, 50, 5, key="dg_qubits")
        
        if circuit_type == "Random Clifford":
            depth = st.slider("Circuit Depth", 5, 50, 15, key="dg_depth")
        elif circuit_type == "QAOA (MaxCut)":
            depth = st.slider("QAOA Steps (p)", 1, 5, 2, key="dg_qaoa")
        else:
            depth = 5 # default for var
            
        noise_scale = st.slider("Noise Scale Factor (Multiplier)", 0.5, 3.0, 1.0, key="dg_noise")
        apply_twirling = st.checkbox("Apply Pauli Twirling", value=True)
        
    st.markdown("---")
    st.subheader("3. Verification and Output Visualization")
    
    safe_to_run = LiveHardwareProfiler.display_compute_estimate(st, n_qubits, depth, 1)
    
    if st.button("Generate & Visualize Sample Data Pair", type="primary", disabled=not safe_to_run):
        profiler = LiveHardwareProfiler(st.sidebar.empty())
        active_files = ["dashboard_tabs/tab_data_generation.py", "data_gen_advanced.py", "qiskit_aer"]
        profiler.update("Generating Training Data Pair", active_files, 0, 1)
        
        with st.spinner("Simulating..."):
            try:
                # 1. Generate Circuit
                if circuit_type == "Random Clifford":
                    qc, instructions = create_random_clifford_circuit(n_qubits, depth)
                elif circuit_type == "QAOA (MaxCut)":
                    qc, instructions = create_qaoa_circuit(n_qubits, p=depth)
                elif circuit_type == "Variational Ansatz":
                    qc, instructions = create_variational_circuit(n_qubits, 5)
                    
                qc.measure_all()
                profiler.update("Generated Topology", active_files, 0, 1, active_qc=qc)
                
                # 2. Get Ground Truth
                true_val, _ = pipeline.get_ground_truth(qc)
                
                # 3. Get Noisy Prediction (using standard prediction logic to mock x_noisy)
                # In a real training pipeline, we'd extract x_noisy directly
                profiler.update("Simulating Data Tensors", active_files, 0, 1, active_qc=qc)
                pred, _, x_noisy = pipeline.predict(qc, instructions, noise_scale=noise_scale)
                
            except Exception as e:
                st.error(f"Error during simulation: {e}")
                return
        
        profiler.update("Data Generation Complete", [], 1, 1)
                
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Math: Ideal ⟨Z₀⟩ Expectation", f"{true_val:.4f}", help="Calculated exactly via Statevector.")
        with c2:
            err = abs(true_val - x_noisy)
            st.metric("Simulation: Noisy ⟨Z₀⟩ Measurement", f"{x_noisy:.4f}", delta=f"Error: {err:.4f}", delta_color="inverse")
        with c3:
            st.metric("Noise Delta", f"{(x_noisy - true_val):.4f}")
            
        st.markdown(f"**This pairs becomes:** `X_input = [x_noisy, n_qubits, depth, {noise_scale}]` -> `Y_target = {true_val:.4f}`")
        
        st.markdown("#### Physical Circuit Visualization")
        # Need to avoid heavy matplotlib rendering locking the main thread if possible, but for Streamlit it's okay
        fig, ax = plt.subplots(figsize=(10, min(n_qubits, 4)))
        ax.axis('off')
        try:
            fig = qc.draw('mpl', ax=ax)
            st.pyplot(fig)
        except:
            st.text(qc.draw('text'))
