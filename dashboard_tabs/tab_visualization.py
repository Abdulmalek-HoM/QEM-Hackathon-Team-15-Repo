import streamlit as st
import utils
from backend.pipeline import HackathonPipeline
import matplotlib.pyplot as plt
import qiskit
from visualization_helpers import plot_error_by_qubit, plot_prediction_scatter, plot_connectivity_heatmap, plot_circuit_dag

def render_visualization_tab(pipeline: HackathonPipeline):
    st.header("🔭 Live Network Visualization & Integrity")
    st.info("This tab ties the dashboard together by actively visualizing the circuit currently selected in the **Live Inference Test** sidebar. It proves the structural integrity of the pipeline by exposing the exact data structures the Graph Transformer is parsing.")
    
    # Safely get session state from sidebar
    circ_type = st.session_state.get("sb_circ", "QAOA (MaxCut)")
    n_qubits = st.session_state.get("sb_q", 4)
    depth = st.session_state.get("sb_d", 2)
    noise = st.session_state.get("sb_n", 1.0)
    
    st.markdown(f"**Currently Observing Sidebar Configuration:** `{circ_type}` with `{n_qubits}` Qubits")
    
    st.markdown("---")
    
    # 1. Re-generate the exact circuit from the sidebar
    try:
        if circ_type == "Random Clifford":
            qc, instructions = utils.create_random_clifford_circuit(n_qubits, depth)
        elif circ_type == "QAOA (MaxCut)":
            qc, instructions = utils.create_qaoa_circuit(n_qubits, p=depth)
        else:
            qc, instructions = utils.create_variational_circuit(n_qubits, depth)
        qc.measure_all()
    except Exception as e:
        st.error(f"Failed to generate circuit. Please adjust parameters. {e}")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Circuit as a Directed Acyclic Graph (DAG)")
        st.markdown("This is the exact topological structure parsed by the `schemas.py` Graph Builder and fed into `CDRFormer`.")
        try:
            fig_dag, nodes, edges = plot_circuit_dag(qc)
            st.pyplot(fig_dag)
            st.caption(f"Graph Metrics: **{nodes} Nodes** | **{edges} Edges**")
        except Exception as e:
            st.warning("DAG graph too dense to render beautifully, or missing dependencies.")
            
        st.subheader("3. Per-Qubit Error Distribution")
        st.markdown(f"Simulating the physical manifestation of `Noise Scale: {noise}x` across the qubit register before and after ML mitigation.")
        fig_err = plot_error_by_qubit(n_qubits, noise)
        st.pyplot(fig_err)

    with col2:
        st.subheader("2. Qubit Connectivity & Hardware Interaction")
        st.markdown("Heatmap showing how often qubits interact. Dense areas represent high crosstalk error zones on physical hardware.")
        fig_heat = plot_connectivity_heatmap(qc)
        st.pyplot(fig_heat)

        st.subheader("4. Architecture Convergence (Scatter Plot)")
        st.markdown("Random batch sampling of 20 analogous circuits, comparing the `CDRFormer` Multi-Layer Perceptron predictions against the exact ideal Physics.")
        fig_scatter = plot_prediction_scatter(pipeline, n_samples=20)
        st.pyplot(fig_scatter)
        
    st.markdown("---")
    st.subheader("5. Raw QASM Circuit Viewer")
    st.markdown("The quantum assembly (QASM) logic underlying the visualizations above.")
    
    with st.expander("View Circuit Graphic Diagram", expanded=True):
        fig, ax = plt.subplots(figsize=(10, min(n_qubits, 6)))
        ax.axis('off')
        try:
            fig = qc.draw('mpl', ax=ax)
            st.pyplot(fig)
        except:
            st.text(qc.draw('text'))
            
    with st.expander("View QASM Source Code"):
        try:
            st.code(qc.qasm(), language="qasm")
        except:
            st.write("QASM parsing unavailable for this parameterized circuit.")
