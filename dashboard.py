import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import os
import json

# Add root to path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.pipeline import HackathonPipeline
from utils import LiveHardwareProfiler
import utils
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
import networkx as nx

# Page Config
st.set_page_config(
    page_title="CDRFormer: Quantum Error Mitigation",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium feel
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #0f3460;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #EEEEEE;
        font-size: 1.1em;
        margin-top: 5px;
    }
    .success-badge {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 0.85em;
    }
    .warning-badge {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 0.85em;
    }
    .info-box {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00d2ff;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        border-top: 1px solid #333;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Load Benchmark Results
@st.cache_data
def load_benchmark_results():
    try:
        with open("assets/benchmark_results.json", "r") as f:
            return json.load(f)
    except:
        return None

# Initialize Pipeline
@st.cache_resource
def load_pipeline():
    return HackathonPipeline(model_path="weights/cdr_former.pth")

pipeline = load_pipeline()
benchmark_data = load_benchmark_results()

# Visualization helpers moved to `visualization_helpers.py`

# --- Sidebar ---
st.sidebar.title("🎛️ Live Inference Test")

# Model Info
st.sidebar.markdown("### 🤖 Model: CDRFormer")
st.sidebar.caption("Graph Transformer with Global Self-Attention")
st.sidebar.markdown("---")

st.sidebar.markdown("Test the underlying `pipeline.py` API right here.")

# Circuit Selection
sb_circuit_type = st.sidebar.selectbox(
    "Select Circuit",
    ["QAOA (MaxCut)", "Random Clifford", "Variational Ansatz"],
    key="sb_circ"
)

# Parameters based on selection
sb_qubits = st.sidebar.slider("Qubits", 2, 50, 4, key="sb_q")
sb_depth = st.sidebar.slider("Depth/Steps", 1, 10, 2, key="sb_d")
sb_noise = st.sidebar.slider("Noise Scale", 0.5, 3.0, 1.0, key="sb_n")

# Pre-computation safety calculator
st.sidebar.markdown("---")
safe_to_run = utils.LiveHardwareProfiler.display_compute_estimate(st.sidebar, sb_qubits, sb_depth)

# Run Button
if st.sidebar.button("🚀 Run Live Inference", type="primary", use_container_width=True, disabled=not safe_to_run):
    with st.sidebar.status("Running Pipeline...", expanded=True) as status:
        try:
            profiler_container = st.sidebar.empty()
            profiler = LiveHardwareProfiler(profiler_container)
            active_files = ["dashboard.py", "backend/pipeline.py", "models/cdr_former.py"]
            
            st.write("Generating circuit...")
            profiler.update("Generating Quantum Context", active_files, 0, 1)
            
            if sb_circuit_type == "Random Clifford":
                qc, instructions = utils.create_random_clifford_circuit(sb_qubits, sb_depth)
            elif sb_circuit_type == "QAOA (MaxCut)":
                qc, instructions = utils.create_qaoa_circuit(sb_qubits, p=sb_depth)
            else:
                qc, instructions = utils.create_variational_circuit(sb_qubits, sb_depth)
            qc.measure_all()
            
            st.write("Calculating physics...")
            profiler.update("Simulating Ground Truth (Density Matrix)", active_files, 0, 1, active_qc=qc)
            true_val, _ = pipeline.get_ground_truth(qc)
            
            st.write("Executing Graph Transformer...")
            profiler.update("Neural Tensor Execution", active_files, 0, 1, active_qc=qc)
            pred, _, zne_base = pipeline.predict(qc, instructions, noise_scale=sb_noise)
            
            status.update(label="Inference Complete!", state="complete", expanded=True)
            profiler.update("Inference Complete", [], 1, 1)
            
            st.sidebar.metric(r"Exact Math Ideal $\langle Z_0 \rangle$", f"{true_val:.4f}")
            st.sidebar.metric("CDRFormer Prediction", f"{pred:.4f}", delta=f"Error: {abs(pred - true_val):.4f}", delta_color="inverse")
            st.sidebar.metric("ZNE Physics Baseline", f"{zne_base:.4f}", delta=f"Error: {abs(zne_base - true_val):.4f}", delta_color="inverse")
            
        except Exception as e:
            status.update(label="Error Occurred", state="error", expanded=True)
            st.sidebar.error(f"{e}")

# --- Main Content ---
st.title("⚛️ CDRFormer Dashboard")
st.markdown("### Data-Driven Quantum Error Mitigation with Graph Transformers")

# Description
st.markdown("""
<div class="info-box">
<strong>Architecture:</strong> CDRFormer combines Clifford Data Regression (CDR) for ground truth generation, 
Pauli Twirling for noise stochastification, and a Graph Transformer that captures circuit topology as a DAG.
<br><br>
<strong>Key Results:</strong> 31.9% error reduction on Variational circuits | 80% win rate vs noisy baselines
</div>
""", unsafe_allow_html=True)

from dashboard_tabs.tab_education import render_education_tab
from dashboard_tabs.tab_data_generation import render_data_generation_tab
from dashboard_tabs.tab_pipeline import render_pipeline_tab
from dashboard_tabs.tab_scripts import render_scripts_tab
from dashboard_tabs.tab_results import render_results_tab
from dashboard_tabs.tab_custom import render_custom_benchmarks_tab
from dashboard_tabs.tab_codebase import render_codebase_tab
from dashboard_tabs.tab_visualization import render_visualization_tab

tab_edu, tab_data, tab_pipe, tab_scripts, tab_results, tab_custom, tab_visuals, tab_codebase = st.tabs([
    "🎓 0. Education", 
    "🧬 1. Data Generation", 
    "⚙️ 2. Pipeline Description", 
    "📜 3. Experiment Scripts", 
    "📈 4. Results", 
    "🧪 5. Custom Benchmarks",
    "🔭 6. Network Visualization",
    "🗂️ 7. Codebase Architecture"
])

with tab_edu:
    render_education_tab()

with tab_data:
    render_data_generation_tab(pipeline)
    
with tab_pipe:
    render_pipeline_tab()
    
with tab_scripts:
    render_scripts_tab(pipeline)
    
with tab_results:
    render_results_tab()
    
with tab_custom:
    render_custom_benchmarks_tab(pipeline)

with tab_visuals:
    render_visualization_tab(pipeline)

with tab_codebase:
    render_codebase_tab()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>Team 15 - The De-Noisers</strong> — Nakahosa Dinovic, Favour Idowu, Abdulmalek Baitulmal<br>
    <em>Hack the Horizon Hackathon</em> • Hosted by African Quantum Consortium<br>
    <a href="https://github.com/Abdulmalek-HoM/QEM-Hackathon-Team-15-Repo" target="_blank">🔗 GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
