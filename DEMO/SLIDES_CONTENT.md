# QEM-Former Presentation Slides
## Team 15 - Hack the Horizon Hackathon

---

# SLIDE 1: TITLE

**Data-Driven Quantum Error Mitigation**
**AI-Assisted Graph Transformer Approach**

🔬 Team 15

- **Nakahosa Dinovic** - Resources Research, Reporter
- **Favour Idowu** - Validation Reviewer, Debugger  
- **Abdulmalek Baitulmal** - Mentor, Solutions Integration

*Hack the Horizon Hackathon - African Quantum Consortium*

---

# SLIDE 2: THE PROBLEM

## NISQ Devices Are Noisy

```
Ideal:  ⟨O⟩ = Tr[Oρ]
Noisy:  ⟨O⟩ = Tr[O·𝒩(ρ)]  ← Corrupted!
```

**Error Sources:**
- 🌡️ Thermal relaxation (T1/T2 decay)
- 📊 Readout errors
- 🔗 Crosstalk between qubits

**Our Goal:**
> Learn a function f_θ that maps noisy → ideal

---

# SLIDE 3: OUR SOLUTION OVERVIEW

## QEM-Former Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  CDR + Pauli    │ ──▶ │   QEM-Former     │ ──▶ │   Benchmarking  │
│  Data Generation│     │  Graph Transformer│     │   & Validation  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**Three Key Innovations:**
1. 📊 **CDR** - Efficient training data generation
2. 🔀 **Pauli Twirling** - Stochastic noise conversion
3. 🕸️ **Graph Transformer** - Topology-aware architecture

---

# SLIDE 4: DATA GENERATION

## Clifford Data Regression (CDR)

**The Problem:** Computing ideal values costs O(2^n) - exponential!

**The Solution:** Clifford circuits simulate in polynomial time

```python
# CDR Algorithm
for i in range(N_samples):
    circuit = RandomCliffordCircuit(n_qubits, depth)
    circuit_twirled = PauliTwirl(circuit)
    y_ideal = StabilizerSimulation(circuit)  # O(poly(n)) ✓
    x_noisy = NoisySimulation(circuit_twirled)
    save(circuit → Graph, x_noisy, y_ideal)
```

**Dataset Statistics:**
| Component | Samples | Circuit Type |
|-----------|---------|--------------|
| Clifford | 5,010 | Random Clifford |
| Mixed | 2,000 | 60% Clifford, 20% QAOA, 20% VQE |
| **Total** | **7,010** | Mixed |

---

# SLIDE 5: PAULI TWIRLING

## Converting Coherent → Stochastic Errors

**Before Twirling:**
```
CNOT → Coherent over-rotation (hard to learn)
```

**After Twirling:**
```
P_c ⊗ P_t · CNOT · P'_c ⊗ P'_t → Stochastic Pauli channel (learnable!)
```

Where P ∈ {I, X, Y, Z} are random Paulis

**Why It Matters:**
- Neural networks learn stochastic patterns better
- Noise becomes predictable across ensemble

---

# SLIDE 6: QEM-FORMER ARCHITECTURE

## Graph Transformer for Quantum Circuits

```
┌─────────────────────────────────────────────────────────────────┐
│                     QEM-Former Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐             │
│  │ Circuit │───▶│ Node Embed   │───▶│TransformerConv│            │
│  │  DAG    │    │ (64-dim)     │    │   (×2)      │            │
│  └─────────┘    └──────────────┘    └──────┬──────┘            │
│                                            │                    │
│  ┌──────────────────────────────────────────┘                    │
│  ▼                                                               │
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────┐      │
│  │ Global Pool   │───▶│ Context Fusion│───▶│ MLP Head    │      │
│  │               │    │               │    │ (128→64→1)  │      │
│  └───────────────┘    └───────┬───────┘    └──────┬──────┘      │
│                               │                    │             │
│                    ┌──────────┴──────────┐         ▼             │
│                    │ [z0_noisy, zz_noisy,│    ⟨Z₀⟩_ideal        │
│                    │  n_qubits, depth,   │                       │
│                    │  noise_scale]       │                       │
│                    └─────────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why Graphs?**
- Circuits ARE Directed Acyclic Graphs
- Nodes = Gates (H, CNOT, RZ...)
- Edges = Qubit wire connections

---

# SLIDE 7: NOISE MODEL

## Thermal Relaxation + Readout Errors

**T1/T2 Relaxation:**
```
p_reset = 1 - e^(-t_g/T1)
p_z = ½(1 - e^(-t_g/T2))(1 - p_reset)
```

**Our Parameters:**
| Parameter | Value |
|-----------|-------|
| T1 | 50 μs (baseline) |
| T2 | 70 μs (baseline) |
| 1-qubit gate | 50 ns |
| 2-qubit gate | 400 ns |
| Readout error | 5% |

**Why Simulated (Not IBM Hardware)?**
✅ Reproducibility  
✅ Controlled experiments  
✅ Exact ground truth via statevector

---

# SLIDE 8: TRAINING

## Hyperparameters & Convergence

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 → 0.000004 |
| LR Scheduler | ReduceLROnPlateau |
| Batch Size | 32 |
| Epochs | 100 |
| Best Epoch | 22 |

**Training Dynamics:**
- Loss: 0.24 → 0.02 in first 10 epochs
- Best validation loss: **0.0094**
- LR decayed 5 times
- No catastrophic overfitting

---

# SLIDE 9: RESULTS - SUCCESS

## Benchmark Results: What Worked

| Circuit Type | Win Rate | Error Reduction | Mean IR |
|--------------|----------|-----------------|---------|
| **Variational** | **80%** ✅ | **31.9%** | 1.44x |
| **Clifford** | 66.7% ✅ | 31.2% | 1.40x |

**Key Metrics:**
- **Win Rate**: How often QEM-Former beats raw noisy measurement
- **Improvement Ratio**: Error_noisy / Error_QEM (>1 is good)

**Interpretation:**
> Our model successfully generalizes from mostly-Clifford training to unseen Variational circuits!

---

# SLIDE 10: RESULTS - FAILURE

## Honest Failure Analysis: QAOA

| Circuit Type | Win Rate | Error Reduction |
|--------------|----------|-----------------|
| QAOA | **15%** ❌ | **-115%** |

**Root Cause:**
- QAOA ideal values ≈ 0
- Model trained on Clifford (values = ±1)
- Model "corrects" toward ±0.05, increasing error

**Proposed Solutions:**
1. Increase QAOA training proportion (8% → 30%)
2. Add circuit-type embedding
3. Separate prediction heads by observable range
4. Uncertainty quantification to abstain on low-confidence

---

# SLIDE 11: ARCHITECTURE EVOLUTION

## Why QEM-Former Was the Right Choice

| Model | Val MSE | Topology-Aware | Noise-Aware |
|-------|---------|----------------|-------------|
| SVR (baseline) | 0.03 | ❌ | ❌ |
| LSTM | 0.03 | Partial | ❌ |
| GCN | 0.02 | ✅ | ❌ |
| **QEM-Former** | **0.009** | ✅ | ✅ |

**Result:** QEM-Former achieves **3.3x better MSE** than baselines

**Key Insight:**
> Circuit topology matters. MLPs and LSTMs ignore it; Graph Transformers capture it.

---

# SLIDE 12: DEMO PREVIEW

## Interactive Streamlit Dashboard

**Features:**
- 📈 View benchmark results
- ⚙️ Adjust noise parameters
- 🔄 Real-time predictions
- 📊 Compare QEM vs ZNE vs Noisy

```bash
# Launch the demo
streamlit run dashboard.py
```

*[Live demo follows]*

---

# SLIDE 13: IMPACT & SCALABILITY

## Beyond the Hackathon

**Scalability Potential:**
- Graph representation → extensible to 100+ qubits
- Transfer learning from small circuits
- Circuit knitting for distributed simulation

**Industry Relevance:**
- Complements hardware error correction
- No additional quantum resources needed
- Applicable to IBM, Google, Amazon quantum systems

**African Development:**
- Quantum technology: frontier for African leadership
- AI + Quantum = high-impact intersection
- This work: proof African teams can contribute cutting-edge research

---

# SLIDE 14: SUMMARY

## Key Contributions

1. ✅ **Complete Pipeline** - CDR + Pauli Twirling + QEM-Former
2. ✅ **31.9% Error Reduction** on Variational circuits
3. ✅ **80% Win Rate** vs noisy baselines
4. ✅ **Transparent Failure Analysis** - QAOA limitations documented
5. ✅ **Reproducible Codebase** with Streamlit dashboard

**What Makes This Unique:**
- Graph representation of quantum circuits
- Noise context injection
- Multi-observable support
- Honest scientific reporting

---

# SLIDE 15: THANK YOU

## Team 15

- **Nakahosa Dinovic** - Resources Research, Reporter
- **Favour Idowu** - Validation Reviewer, Debugger
- **Abdulmalek Baitulmal** - Mentor, Solutions Integration

**Repository:** [github.com/Abdulmalek-HoM/QEM-Hackathon-Team-15-Repo](https://github.com/Abdulmalek-HoM/QEM-Hackathon-Team-15-Repo)

**Built with AI-Assisted Research** 🤖

*Thank you, African Quantum Consortium!*

---

# BACKUP SLIDES

## B1: Q&A - Why Simulated Noise?

**Alternative:** Use real IBM hardware noise

**Our Choice:** Simulated with T1/T2 physics

**Reasons:**
| Factor | Simulated | Hardware |
|--------|-----------|----------|
| Reproducibility | ✅ Perfect | Varies daily |
| Ground Truth | ✅ Statevector | Needs tomography |
| Scientific Control | ✅ Isolated | Coupled effects |

---

## B2: Q&A - How to Scale to 100+ Qubits?

1. **Circuit Knitting** - Distribute simulation
2. **Tensor Networks** - Approximate ground truth
3. **Transfer Learning** - Train on small, apply to large
4. **Approximate Methods** - MPS, DMRG for validation

---

## B3: Code Snippets

**Data Generation:**
```python
python data_gen_advanced.py --large
python data_gen_advanced.py --mixed --samples 2000
```

**Training:**
```python
python train_qem.py
```

**Benchmarking:**
```python
python benchmark_suite.py
```
