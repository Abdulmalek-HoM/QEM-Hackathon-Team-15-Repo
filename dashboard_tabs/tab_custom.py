import streamlit as st
import pandas as pd
import numpy as np
import time
from backend.pipeline import HackathonPipeline
from utils import create_random_clifford_circuit, create_qaoa_circuit, create_variational_circuit, LiveHardwareProfiler
import plotly.express as px

def render_custom_benchmarks_tab(pipeline: HackathonPipeline):
    st.header("🧪 Custom Benchmarks")
    st.info("Run flexible, customizable simulations to test CDRFormer against your own parameters.")
    
    st.markdown("### 1. Configure Evaluation Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_type = st.selectbox("Pipeline Target", ["CDRFormer (Graph Transformer)", "ZNE (Zero Noise Extrapolation)", "Both"])
    with col2:
        circuit_mix = st.multiselect("Validation Data Sources", ["Random Clifford", "QAOA (MaxCut)", "Variational Ansatz"], default=["QAOA (MaxCut)"])
    with col3:
        samples = st.number_input("Number of Benchmark Samples", min_value=1, max_value=50, value=5)
    
    max_qubits = st.slider("Maximum Qubits in Custom Generation", 2, 50, 6)

    st.markdown("---")
    
    st.markdown("### 2. Computing Safety Check")
    # Custom Benchmark simulates multiple types, so depth averages ~15
    total_samples = len(circuit_mix) * samples if circuit_mix else 0
    safe_to_run = LiveHardwareProfiler.display_compute_estimate(st, max_qubits, 15, total_samples)
    
    st.markdown("### 3. Execution Engine")
    
    if st.button("🚀 Execute Custom Benchmark", type="primary", disabled=not safe_to_run):
        if not circuit_mix:
            st.warning("Please select at least one circuit type to validate against.")
            return
            
        st.write("Initializing Quantum Environment...")
        
        progress_bar = st.progress(0)
        results = []
        
        total_runs = len(circuit_mix) * samples
        current_run = 0
        
        # Initialize sidebar profiler
        profiler = LiveHardwareProfiler(st.sidebar.empty())
        active_files = ["dashboard_tabs/tab_custom.py", "backend/pipeline.py", "qiskit_aer"]
        
        for c_type in circuit_mix:
            for s in range(samples):
                profiler.update(f"Simulating {c_type}", active_files, current_run, total_runs)
                try:
                    # 1. Generate Circuit
                    n_qubits = np.random.randint(2, max_qubits + 1)
                    if c_type == "Random Clifford":
                        depth = np.random.randint(5, 20)
                        qc, instructions = create_random_clifford_circuit(n_qubits, depth)
                    elif c_type == "QAOA (MaxCut)":
                        p_steps = np.random.randint(1, 4)
                        qc, instructions = create_qaoa_circuit(n_qubits, p=p_steps)
                    elif c_type == "Variational Ansatz":
                        qc, instructions = create_variational_circuit(n_qubits, 5)
                        
                    qc.measure_all()

                    # 2. Run Pipeline
                    pred, ai_res, zne_base = pipeline.predict(qc, instructions)
                    profiler.update(f"Analyzing {c_type}", active_files, current_run, total_runs, active_qc=qc)
                    
                    # 3. Get Ground Truth
                    true_val, _ = pipeline.get_ground_truth(qc)
                    
                    # 4. Metrics
                    err_base = abs(true_val - zne_base)
                    err_model = abs(true_val - pred)
                    winner = "CDRFormer" if err_model < err_base else "ZNE"
                    
                    results.append({
                        "Circuit Type": c_type,
                        "Qubits": n_qubits,
                        "Ideal ⟨Z₀⟩": round(true_val, 4),
                        "ZNE ⟨Z₀⟩": round(zne_base, 4),
                        "CDRFormer ⟨Z₀⟩": round(pred, 4),
                        "ZNE Error": round(err_base, 4),
                        "CDR Error": round(err_model, 4),
                        "Winner": winner
                    })
                except Exception as e:
                    pass
                
                current_run += 1
                progress_bar.progress(current_run / total_runs)
                profiler.update(f"Simulating {c_type}", active_files, current_run, total_runs)
                
        st.success("✅ Custom Benchmark Complete!")
        profiler.update("Benchmark Complete", [], current_run, total_runs)
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Win rates & Visualizations
        if len(df) > 0:
            cdr_wins = len(df[df["Winner"] == "CDRFormer"])
            st.metric("Custom CDRFormer Win Rate", f"{(cdr_wins/len(df))*100:.1f}%")
            
            st.markdown("### 📊 Benchmark Performance Analysis")
            # Group by circuit type to show average accuracy visually
            avg_df = df.groupby("Circuit Type")[["ZNE Error", "CDR Error"]].mean().reset_index()
            # Melt for easy plotly grouping
            melted_df = pd.melt(avg_df, id_vars=['Circuit Type'], value_vars=['ZNE Error', 'CDR Error'], 
                                var_name='Mitigation Method', value_name='Average Error Magnitude')
                                
            fig = px.bar(melted_df, x='Circuit Type', y='Average Error Magnitude', color='Mitigation Method', barmode='group',
                         title='Average Absolute Error by Circuit Topology (Lower is Better)',
                         color_discrete_map={'ZNE Error': '#F39C12', 'CDR Error': '#2ECC71'},
                         template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        st.download_button(
            label="📥 Download Custom Results (CSV)",
            data=df.to_csv(index=False),
            file_name="custom_benchmark_results.csv",
            mime="text/csv"
        )
