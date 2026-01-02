import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import os

# Add root to path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.pipeline import HackathonPipeline
import utils
from qiskit import QuantumCircuit

# Page Config
st.set_page_config(
    page_title="QEM: Quantum Error Mitigation",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #262730;
        text-align: center;
    }
    .metric-value {
        font-size: 3em;
        font-weight: bold;
        color: #00ADB5;
    }
    .metric-label {
        color: #EEEEEE;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Pipeline
@st.cache_resource
def load_pipeline():
    return HackathonPipeline(lstm_path="qem_lstm.pth")

pipeline = load_pipeline()

# --- Sidebar ---
st.sidebar.title("🎛️ Control Panel")

# Circuit Selection
circuit_type = st.sidebar.selectbox(
    "Select Circuit Type",
    ["Random Clifford", "QAOA (MaxCut)", "Variational Ansatz", "Custom QASM"]
)

# Parameters based on selection
depth = 10
qubits = 2 # Fixed for this demo as per models
if circuit_type == "Random Clifford":
    depth = st.sidebar.slider("Circuit Depth", 5, 50, 15)
elif circuit_type == "QAOA (MaxCut)":
    p_steps = st.sidebar.slider("QAOA Steps (p)", 1, 5, 2)

# Noise Settings (Simulated)
st.sidebar.markdown("---")
st.sidebar.subheader("noise Settings")
noise_scale = st.sidebar.slider("Noise Scale Factor", 0.5, 3.0, 1.0)

# Run Button
run_btn = st.sidebar.button("🚀 Run Mitigation", type="primary")

# --- Main Content ---
st.title("⚛️ Quantum Error Mitigation (QEM) Dashboard")
st.markdown("### Hybrid Architecture: Zero-Noise Extrapolation + AI Correction")

if run_btn:
    with st.spinner("Compiling Circuit & Running Simulations..."):
        # 1. Generate Circuit
        if circuit_type == "Random Clifford":
            qc, instructions = utils.create_random_clifford_circuit(qubits, depth)
        elif circuit_type == "QAOA (MaxCut)":
            qc, instructions = utils.create_qaoa_circuit(qubits, p=p_steps)
        elif circuit_type == "Variational Ansatz":
            qc, instructions = utils.create_variational_circuit(qubits, 5)
        else:
            # Placeholder for Custom QASM
            qc, instructions = utils.create_random_clifford_circuit(qubits, 10)
            st.warning("using random circuit for now")

        # 2. Run Pipeline
        final_pred, ai_res, zne_base = pipeline.predict(qc, instructions)
        
        # 3. Get Ground Truth
        true_val, counts_ideal = pipeline.get_ground_truth(qc)
        
        # 4. Metrics
        err_base = abs(true_val - zne_base)
        err_model = abs(true_val - final_pred)
        ratio = 100.0 if err_model < 1e-5 else err_base / err_model
        
    # --- Results View ---
    
    # Kpi Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Expected Value (True)", value=f"{true_val:.4f}")
    with col2:
        st.metric(label="ZNE Baseline", value=f"{zne_base:.4f}", delta=f"-{err_base:.4f} Err", delta_color="inverse")
    with col3:
        st.metric(label="Hybrid AI Model", value=f"{final_pred:.4f}", delta=f"-{err_model:.4f} Err", delta_color="inverse")
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Improvement Ratio</div>
            <div class="metric-value">{ratio:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)

    # Visualization Row
    st.markdown("### 📊 Performance Analysis")
    
    tab1, tab2 = st.tabs(["Comparison", "Circuit Viewer"])
    
    with tab1:
        # Comparison Chart
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # Bar Chart: True vs ZNE vs AI
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = ['True', 'ZNE (Phys)', 'Hybrid (Phys+AI)']
            values = [true_val, zne_base, final_pred]
            colors = ['#2ECC71', '#E74C3C', '#3498DB']
            
            bars = ax.bar(labels, values, color=colors, alpha=0.8)
            ax.axhline(y=true_val, color='green', linestyle='--', alpha=0.5, label="Target")
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom')
            
            ax.set_ylabel("Expectation Value <Z>")
            ax.set_title("Mitigation Accuracy Comparison")
            st.pyplot(fig)
            
        with c2:
            st.markdown("#### AI contribution")
            st.info(f"The AI Model predicted a residual correction of: **{ai_res:.5f}**")
            st.markdown("""
            This value accounts for complex non-linear errors that standard scaling (ZNE) missed, 
            such as cross-talk and pattern-dependent noise.
            """)

    with tab2:
        st.pyplot(qc.draw('mpl'))

else:
    st.info("👈 Select a circuit type and click 'Run Mitigation' to start.")
