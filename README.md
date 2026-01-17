<div align="center">

# 🏆 QEM-Former: Data-Driven Quantum Error Mitigation

### **🎉 Innovation Award Winner — Hack the Horizon 2025 🎉**

*Awarded for standout originality and a clear technical leap beyond the baseline*

![The De-Noisers Team](QEM-FORMER%20The%20De-Noisers%20Team%2015.png)

**Team 15 — The De-Noisers**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Overview

QEM-Former is a novel **Graph Transformer architecture** for quantum error mitigation that learns to correct noisy quantum measurements using machine learning. Our approach combines **Clifford Data Regression (CDR)** for efficient training data generation with **Pauli Twirling** for noise stochastification.

<div align="center">

![The Challenge](Slides/figures/slide_02_problem_v2_1768513497052.png)

</div>

### Key Achievements
- 🎯 **31.9% Error Reduction** on Variational circuits
- 🏆 **80% Win Rate** against raw noisy execution
- 📊 **7,010 Training Samples** generated via CDR
- 🔬 **3.3x Better** than baseline architectures (SVR, LSTM, GCN)

---

## 🧪 The Problem We Solve

When we measure a quantum observable, NISQ devices introduce errors through thermal relaxation, dephasing, and readout noise. Our goal is to learn a function that maps noisy measurements back to ideal values.

```
⟨O⟩_ideal = Tr[Oρ]  →  ⟨O⟩_noisy = Tr[O·N(ρ)]  →  QEM-Former  →  ⟨O⟩_predicted ≈ ⟨O⟩_ideal
```

---

## 🔬 Methodology

### Phase 1: Data Generation (CDR + Pauli Twirling)

<div align="center">

![Data Pipeline](Slides/figures/slide_03_data_pipeline_v2_1768513527742.png)

</div>

- **Clifford Data Regression**: Efficiently simulable circuits provide exact ground truth via stabilizer formalism
- **Pauli Twirling**: Random Pauli insertions around CNOTs convert coherent → stochastic errors
- **Multi-Observable Support**: Predicts `⟨Z₀⟩`, `⟨Z₀Z₁⟩` correlations, and global parity

<div align="center">

![CDR & Pauli Twirling Details](Slides/figures/cdr_pauli_twirling_1768503241552.png)

</div>

### Phase 2: QEM-Former Architecture

<div align="center">

![QEM-Former Architecture](Slides/figures/slide_04_architecture_v2_1768513580405.png)

</div>

Our Graph Transformer captures circuit topology as a Directed Acyclic Graph (DAG):

1. **Node Embedding**: Gate types (H, CNOT, RZ) → 64-dim learned vectors
2. **TransformerConv Layers**: 2 graph attention layers capture local topology
3. **Global Pooling**: Aggregate node features into graph-level embedding
4. **Context Fusion**: Inject noise information (noisy measurement, qubit count, depth, noise scale)
5. **Regression Head**: 3-layer MLP outputs predicted ideal expectation value

### Phase 3: Results

<div align="center">

![Benchmark Results](Slides/figures/slide_05_results_v2_1768513640613.png)

</div>

| Circuit Type | Win Rate | Error Reduction | Improvement Ratio |
|--------------|----------|-----------------|-------------------|
| **Variational** | **80%** | **31.9%** | **1.47x** |
| Clifford | 66.7% | 31.2% | 1.45x |
| QAOA | 15% | -115% | 0.46x |

### Architecture Evolution

<div align="center">

![Architecture Comparison](Slides/figures/slide_07_arch_comp_v2_1768513715304.png)

</div>

We iteratively improved through 4 architectures, achieving **3.3x better MSE** with QEM-Former.

---

## ⚠️ Honest Failure Analysis

<div align="center">

![QAOA Failure Analysis](Slides/figures/slide_06_qaoa_failure_v2_1768513687323.png)

</div>

We believe in transparency. QAOA circuits underperformed (15% win rate) because:
- QAOA ideal values cluster near zero
- Model trained primarily on Clifford data (values at ±1, 0)
- **Fix**: Increase QAOA training proportion from 8% to 30%+

---

## 🖥️ Interactive Dashboard

Our Streamlit dashboard provides real-time quantum error mitigation with visualization:

<div align="center">

### Live Demo Mode
![Dashboard Overview](Dashboard%201.png)

### Benchmark Results
![Dashboard Benchmarks](Dashboard%202.png)

### Presentation Mode
![Dashboard Presentation](Dashboard%203.png)

</div>

**Features:**
- 🔬 Live circuit simulation and mitigation
- 📊 Error Distribution by Qubit visualization
- 🔗 Qubit Connectivity Heatmap
- 🎬 **Presentation Mode** — Step-by-step demo aligned with our presentation

```bash
# Launch the dashboard
streamlit run dashboard.py
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Abdulmalek-HoM/QEM-Hackathon-Team-15-Repo.git
cd QEM-Hackathon-Team-15-Repo

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# Generate training data (500 samples for quick start)
python data_gen_advanced.py --samples 500

# Generate large dataset (5000+ samples)
python data_gen_advanced.py --large

# Train the model
python train_qem.py

# Run benchmarks (includes OOD testing)
python benchmark_suite.py

# Launch interactive dashboard
streamlit run dashboard.py
```

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `data_gen_advanced.py` | CDR + Pauli Twirling data generation |
| `models/qem_former.py` | Graph Transformer architecture |
| `train_qem.py` | Training with visualization |
| `benchmark_suite.py` | OOD testing + JSON export |
| `dashboard.py` | Streamlit visualization interface |
| `backend/pipeline.py` | Inference pipeline |
| `Slides/figures/` | Presentation slides and diagrams |

---

## 🌍 Impact & Business Value

<div align="center">

![Impact](Slides/figures/slide_09_impact_v2_1768561997099.png)

</div>

- **Scalability**: Graph representation extends to 100+ qubits via circuit knitting
- **Industry Relevance**: Complements hardware improvements at IBM, Google, Amazon
- **African Development**: Demonstrating cutting-edge quantum research from Africa

---

## 📋 Requirements

- Python 3.10+
- Qiskit 1.0+, Qiskit-Aer 0.14+
- PyTorch 2.1+, PyTorch Geometric 2.4+
- Mitiq 0.35+
- Streamlit 1.30+

See `requirements.txt` for complete dependencies.

---

## 👥 Team 15 — The De-Noisers

| Name | Role |
|------|------|
| **Nakahosa Dinovic** | Resources Research |
| **Favour Idowu** | Validation Reviewer |
| **Abdulmalek Baitulmal** | Mentor & Integration |

---

## 🏆 Recognition

**🏅 Innovation Award Winner — Hack the Horizon 2025**

*Hosted by the African Quantum Consortium (AQC)*

> "This work demonstrated standout originality and a clear technical leap beyond the baseline. This is exactly the kind of bold experimentation we want to amplify across the continent." — AQC Operations Team

---

<div align="center">

**Made with ❤️ for the African Quantum Consortium**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Abdulmalek-HoM/QEM-Hackathon-Team-15-Repo)

</div>
