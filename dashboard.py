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
import utils
from qiskit import QuantumCircuit

# Page Config
st.set_page_config(
    page_title="QEM-Former: Quantum Error Mitigation",
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
        with open("benchmark_results.json", "r") as f:
            return json.load(f)
    except:
        return None

# Initialize Pipeline
@st.cache_resource
def load_pipeline():
    return HackathonPipeline(model_path="qem_former.pth")

pipeline = load_pipeline()
benchmark_data = load_benchmark_results()

# --- Sidebar ---
st.sidebar.title("🎛️ Control Panel")

# Model Info
st.sidebar.markdown("### 🤖 Model: QEM-Former")
st.sidebar.caption("Graph Transformer with Global Self-Attention")

st.sidebar.markdown("---")

# Circuit Selection
circuit_type = st.sidebar.selectbox(
    "Select Circuit Type",
    ["Random Clifford", "QAOA (MaxCut)", "Variational Ansatz", "Custom QASM"]
)

# Parameters based on selection
depth = 10
qubits = st.sidebar.slider("Number of Qubits", 2, 10, 5)
if circuit_type == "Random Clifford":
    depth = st.sidebar.slider("Circuit Depth", 5, 50, 15)
elif circuit_type == "QAOA (MaxCut)":
    p_steps = st.sidebar.slider("QAOA Steps (p)", 1, 5, 2)

# Noise Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Noise Settings")
noise_scale = st.sidebar.slider("Noise Scale Factor", 0.5, 3.0, 1.0)

# Noise Model Info
with st.sidebar.expander("📊 Noise Model Details"):
    t1_val = 50e-6 / noise_scale
    t2_val = 70e-6 / noise_scale
    st.markdown(f"""
    **Thermal Relaxation Parameters:**
    - T₁ = {t1_val*1e6:.1f} μs
    - T₂ = {t2_val*1e6:.1f} μs
    - 1Q Gate: 50 ns
    - 2Q Gate: 400 ns
    - Readout Error: {min(50, 5*noise_scale):.1f}%
    """)

# Run Button
run_btn = st.sidebar.button("🚀 Run Mitigation", type="primary")

# --- Main Content ---
st.title("⚛️ QEM-Former Dashboard")
st.markdown("### Data-Driven Quantum Error Mitigation with Graph Transformers")

# Description
st.markdown("""
<div class="info-box">
<strong>Architecture:</strong> QEM-Former combines Clifford Data Regression (CDR) for ground truth generation, 
Pauli Twirling for noise stochastification, and a Graph Transformer that captures circuit topology as a DAG.
<br><br>
<strong>Key Results:</strong> 31.9% error reduction on Variational circuits | 80% win rate vs noisy baselines
</div>
""", unsafe_allow_html=True)

# Main Tabs
main_tab1, main_tab2, main_tab3 = st.tabs(["🔬 Live Demo", "📈 Benchmark Results", "🏗️ Architecture"])

with main_tab1:
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
                st.warning("Custom QASM not implemented — using random Clifford circuit.")
                
            # Ensure measurements are present for simulations
            qc.measure_all()

            # 2. Run Pipeline
            final_pred, ai_res, zne_base = pipeline.predict(qc, instructions)
            
            # 3. Get Ground Truth
            true_val, counts_ideal = pipeline.get_ground_truth(qc)
            
            # 4. Metrics
            err_base = abs(true_val - zne_base)
            err_model = abs(true_val - final_pred)
            ratio = 100.0 if err_model < 1e-5 else err_base / err_model
            
        # --- Results View ---
        st.markdown("### 📊 Results")
        
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Expected Value (True)", value=f"{true_val:.4f}")
        with col2:
            st.metric(label="ZNE Baseline", value=f"{zne_base:.4f}", delta=f"-{err_base:.4f} Err", delta_color="inverse")
        with col3:
            st.metric(label="QEM-Former", value=f"{final_pred:.4f}", delta=f"-{err_model:.4f} Err", delta_color="inverse")
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
                labels = ['True', 'ZNE (Physics)', 'QEM-Former (AI)']
                values = [true_val, zne_base, final_pred]
                colors = ['#2ECC71', '#E74C3C', '#3498DB']
                
                bars = ax.bar(labels, values, color=colors, alpha=0.8)
                ax.axhline(y=true_val, color='green', linestyle='--', alpha=0.5, label="Target")
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom')
                
                ax.set_ylabel("Expectation Value ⟨Z₀⟩")
                ax.set_title("Mitigation Accuracy Comparison")
                ax.set_facecolor('#0E1117')
                fig.patch.set_facecolor('#0E1117')
                ax.tick_params(colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('#333')
                st.pyplot(fig)
                
            with c2:
                st.markdown("#### 🤖 AI Contribution")
                st.info(f"Residual correction: **{ai_res:.5f}**")
                st.markdown("""
                The QEM-Former captures non-linear error patterns that ZNE's polynomial extrapolation misses, 
                including gate-dependent noise and circuit topology effects.
                """)

        with tab2:
            st.pyplot(qc.draw('mpl'))

    else:
        st.info("👈 Select a circuit type and click **'Run Mitigation'** to start the live demo.")

with main_tab2:
    st.markdown("### 📈 Pre-computed Benchmark Results")
    
    if benchmark_data:
        # Summary Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cliff_data = benchmark_data.get("in_distribution", {})
            win_rate = cliff_data.get("win_rate", 0) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Random Clifford</div>
                <div class="metric-value">{win_rate:.0f}%</div>
                <div class="metric-label">Win Rate</div>
                <span class="success-badge">In-Distribution</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            var_data = benchmark_data.get("ood_variational", {})
            win_rate = var_data.get("win_rate", 0) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Variational Ansatz</div>
                <div class="metric-value">{win_rate:.0f}%</div>
                <div class="metric-label">Win Rate</div>
                <span class="success-badge">Best Result</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            qaoa_data = benchmark_data.get("ood_qaoa", {})
            win_rate = qaoa_data.get("win_rate", 0) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">QAOA</div>
                <div class="metric-value">{win_rate:.0f}%</div>
                <div class="metric-label">Win Rate</div>
                <span class="warning-badge">Needs Improvement</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Table
        st.markdown("#### 📋 Detailed Results")
        
        results_df = pd.DataFrame({
            "Circuit Type": ["Random Clifford", "Variational Ansatz", "QAOA"],
            "Win Rate": [
                f"{benchmark_data['in_distribution']['win_rate']*100:.1f}%",
                f"{benchmark_data['ood_variational']['win_rate']*100:.1f}%",
                f"{benchmark_data['ood_qaoa']['win_rate']*100:.1f}%"
            ],
            "Error Reduction": [
                f"{(1 - benchmark_data['in_distribution']['mean_qem_error']/benchmark_data['in_distribution']['mean_noisy_error'])*100:.1f}%",
                f"{(1 - benchmark_data['ood_variational']['mean_qem_error']/benchmark_data['ood_variational']['mean_noisy_error'])*100:.1f}%",
                f"{(1 - benchmark_data['ood_qaoa']['mean_qem_error']/benchmark_data['ood_qaoa']['mean_noisy_error'])*100:.1f}%"
            ],
            "Improvement Ratio": [
                f"{benchmark_data['in_distribution']['mean_ir_qem']:.2f}x",
                f"{benchmark_data['ood_variational']['mean_ir_qem']:.2f}x",
                f"{benchmark_data['ood_qaoa']['mean_ir_qem']:.2f}x"
            ],
            "Mean QEM Error": [
                f"{benchmark_data['in_distribution']['mean_qem_error']:.4f}",
                f"{benchmark_data['ood_variational']['mean_qem_error']:.4f}",
                f"{benchmark_data['ood_qaoa']['mean_qem_error']:.4f}"
            ]
        })
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Visualization
        st.markdown("#### 📊 Error Comparison by Circuit Type")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0E1117')
        
        # Bar chart - Error comparison
        ax1 = axes[0]
        circuit_types = ['Clifford', 'Variational', 'QAOA']
        noisy_errors = [
            benchmark_data['in_distribution']['mean_noisy_error'],
            benchmark_data['ood_variational']['mean_noisy_error'],
            benchmark_data['ood_qaoa']['mean_noisy_error']
        ]
        qem_errors = [
            benchmark_data['in_distribution']['mean_qem_error'],
            benchmark_data['ood_variational']['mean_qem_error'],
            benchmark_data['ood_qaoa']['mean_qem_error']
        ]
        
        x = np.arange(len(circuit_types))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, noisy_errors, width, label='Noisy', color='#E74C3C', alpha=0.8)
        bars2 = ax1.bar(x + width/2, qem_errors, width, label='QEM-Former', color='#3498DB', alpha=0.8)
        
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Error Comparison: Noisy vs QEM-Former')
        ax1.set_xticks(x)
        ax1.set_xticklabels(circuit_types)
        ax1.legend()
        ax1.set_facecolor('#0E1117')
        ax1.tick_params(colors='white')
        ax1.yaxis.label.set_color('white')
        ax1.title.set_color('white')
        for spine in ax1.spines.values():
            spine.set_color('#333')
        ax1.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        
        # Win rate chart
        ax2 = axes[1]
        win_rates = [
            benchmark_data['in_distribution']['win_rate'] * 100,
            benchmark_data['ood_variational']['win_rate'] * 100,
            benchmark_data['ood_qaoa']['win_rate'] * 100
        ]
        colors = ['#3498DB', '#2ECC71', '#E74C3C']
        
        bars = ax2.bar(circuit_types, win_rates, color=colors, alpha=0.8)
        ax2.axhline(y=50, color='yellow', linestyle='--', alpha=0.7, label='Random Chance')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('QEM-Former Win Rate by Circuit Type')
        ax2.set_ylim(0, 100)
        ax2.set_facecolor('#0E1117')
        ax2.tick_params(colors='white')
        ax2.yaxis.label.set_color('white')
        ax2.title.set_color('white')
        for spine in ax2.spines.values():
            spine.set_color('#333')
        ax2.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.0f}%', ha='center', va='bottom', color='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Note
        st.caption(f"📅 Results from: {benchmark_data.get('timestamp', 'N/A')[:10]} | {benchmark_data.get('note', '')}")
        
    else:
        st.warning("Benchmark results not found. Run `python benchmark_suite.py` to generate.")

with main_tab3:
    st.markdown("### 🏗️ QEM-Former Architecture")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        #### Data Flow Pipeline
        
        ```
        Quantum Circuit ──► DAG Representation ──► Graph Transformer ──► Mitigated Value
                │                   │                      │
                │                   ▼                      ▼
                │            Node Embeddings         Global Context
                │            (Gate Types)            [Z₀_noisy, n_qubits, depth]
                │                   │                      │
                │                   ▼                      │
                │            TransformerConv (×2)          │
                │                   │                      │
                │                   ▼                      │
                │            Global Mean Pool              │
                │                   │                      │
                │                   └──────► Fusion ◄──────┘
                │                              │
                │                              ▼
                │                          MLP Head
                │                              │
                │                              ▼
                └────────────────────► ⟨Z₀⟩_ideal (Predicted)
        ```
        
        #### Key Components
        
        1. **Node Embedding**: Gate types → 64-dim vectors
        2. **TransformerConv Layers**: 2 layers with 4 attention heads
        3. **Global Context Fusion**: Noisy measurements + circuit metadata
        4. **Output Regressor**: 3-layer MLP → single expectation value
        """)
    
    with col2:
        st.markdown("""
        #### Training Details
        
        | Parameter | Value |
        |-----------|-------|
        | Optimizer | Adam |
        | Learning Rate | 0.001 |
        | LR Scheduler | ReduceLROnPlateau |
        | Batch Size | 32 |
        | Epochs | 100 |
        | Loss | MSE |
        | Best Val Loss | 0.0094 |
        
        ---
        
        #### Dataset
        
        - **7,010 total samples**
        - 5,010 Clifford circuits
        - 2,000 mixed (Clifford + QAOA + VQE)
        - Ground truth via stabilizer + statevector
        """)
    
    st.markdown("---")
    
    st.markdown("""
    #### Noise Model (Simulated)
    
    Our training uses a physics-based simulated noise model for reproducibility:
    
    | Parameter | Baseline Value | Description |
    |-----------|---------------|-------------|
    | T₁ | 50 μs | Amplitude damping time |
    | T₂ | 70 μs | Dephasing time |
    | 1Q Gate | 50 ns | Single-qubit gate duration |
    | 2Q Gate | 400 ns | Two-qubit gate duration |
    | Readout Error | 5% | Symmetric confusion matrix |
    
    *Noise scale factor adjusts T₁/T₂ inversely and readout error directly.*
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>Team 15</strong> — Nakahosa Dinovic, Favour Idowu, Abdulmalek Baitulmal<br>
    <em>Hack the Horizon Hackathon</em> • Hosted by African Quantum Consortium<br>
    <a href="https://github.com/Abdulmalek-HoM/QEM-Hackathon-Team-15-Repo" target="_blank">🔗 GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
