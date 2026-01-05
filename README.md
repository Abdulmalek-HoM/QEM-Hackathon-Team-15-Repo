# Team 15: Data-Driven Quantum Error Mitigation (QEM)

## Overview
This repository contains Team 15's submission to the **Aqora "Data-Driven Quantum Error Mitigation" Competition**. We develop a robust, data-driven pipeline to mitigate noise in NISQ devices using advanced **Graph Transformers (QEMFormer)** combined with physics-based methods.

## Key Features
- **Multi-Observable Support**: Predicts `⟨Z₀⟩`, `⟨Z₀Z₁⟩` correlations, and global parity
- **Graph Transformer Architecture**: Captures circuit topology as DAG for superior generalization
- **Clifford Data Regression (CDR)**: Overcomes supervision bottleneck with near-Clifford training
- **Pauli Twirling**: Converts coherent errors to stochastic for cleaner learning
- **OOD Testing**: Benchmarks on Clifford, QAOA, and Variational circuits

## Quick Start

```bash
# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# Generate training data (500 samples)
python data_gen_advanced.py --samples 500

# Generate large dataset (5000 samples)
python data_gen_advanced.py --large

# Train the model
python train_qem.py

# Run benchmarks (includes OOD testing)
python benchmark_suite.py

# Launch interactive dashboard
streamlit run dashboard.py
```

## Project Structure

### Core Pipeline
| File | Description |
|------|-------------|
| `data_gen_advanced.py` | CDR + Pauli Twirling data generation with multi-observable support |
| `models/qem_former.py` | Graph Transformer with noise context injection |
| `train_qem.py` | Training with visualization and metrics logging |
| `benchmark_suite.py` | OOD testing across circuit types + JSON export |
| `dashboard.py` | Streamlit visualization interface |

### Educational Modules
Jupyter notebooks (Module 1-7) provide a progressive walkthrough from qubit basics to deployment.

## Methodology

### Phase 1: Data Generation
- **Clifford Data Regression (CDR)**: Efficiently simulable circuits provide ground truth
- **Pauli Twirling**: Random Pauli insertions around CNOTs convert coherent → stochastic errors
- **Variable Noise**: Training at multiple noise scales (configurable `--noise-scale`)

### Phase 2: Model Architecture
```
Circuit DAG → Node Embedding → TransformerConv (×2) → Global Pool → Context Fusion → MLP → ⟨O⟩_ideal
                                                              ↑
                                                    [z0_noisy, zz_noisy, n_qubits, depth, noise_scale]
```

### Phase 3: Benchmarking
- **In-Distribution**: Random Clifford circuits
- **Out-of-Distribution**: QAOA and Variational Ansatz circuits
- **Metrics**: Improvement Ratio (IR) = Error_noisy / Error_mitigated

## Limitations & Failure Cases

| Scenario | Expected Behavior | Mitigation |
|----------|------------------|------------|
| Deep circuits (>50 gates) | Reduced IR due to noise accumulation | Increase training depth range |
| Non-Clifford heavy circuits | Lower accuracy (CDR limitation) | Variable-noise augmentation |
| Strong crosstalk | May underperform if not in training | Include multi-qubit correlations |
| Low shot counts (<100) | Statistical noise dominates | Ensemble predictions |

## Output Files

After running the pipeline:
- `qem_former.pth` — Trained model weights
- `training_metrics.json` — Loss curves and training stats
- `training_curves.png` — Visualization of training progress
- `benchmark_results.json` — Detailed benchmark metrics

## Requirements
- Python 3.10+
- Qiskit 1.0+, Qiskit-Aer 0.14+
- PyTorch 2.1+, PyTorch Geometric 2.4+
- Mitiq 0.35+

See `requirements.txt` for complete dependencies.

## Authors
**Aqora Competition Strategy Team (Team 15)**
