# QEM-Former Video Presentation Script
## Team 15 - Hack the Horizon Hackathon

**Total Duration: 8-10 minutes**

---

## PART 1: INTRODUCTION (1 minute)
*[Screen: Title slide with team names]*

**SPEAKER:**
> "Hello, we're Team 15 from the Hack the Horizon Hackathon. Today we're presenting our solution for data-driven quantum error mitigation using a novel Graph Transformer architecture we call **QEM-Former**.

> The problem we're solving is fundamental to quantum computing: NISQ devices are noisy. Every measurement we take is corrupted by errors from thermal relaxation, dephasing, and readout. Our goal is to learn a correction function that recovers the ideal quantum expectation values from noisy measurements."

---

## PART 2: METHODOLOGY (3 minutes)

### 2.1 The Challenge (30 sec)
*[Screen: Problem equation slide]*

**SPEAKER:**
> "When we measure an observable O on an ideal quantum state ρ, we get the true expectation value. But with noise channel N, we get a corrupted result. The challenge is learning to map noisy → ideal."

**Key Visual:** Show equation:
```
⟨O⟩_ideal = Tr[Oρ]  vs  ⟨O⟩_noisy = Tr[O·N(ρ)]
```

### 2.2 Data Generation - CDR + Pauli Twirling (1 min)
*[Screen: Data pipeline diagram]*

**SPEAKER:**
> "The first innovation is our data generation. Training a neural network requires pairs of noisy and ideal measurements. But computing ideal values for large circuits is exponentially expensive.

> We solve this using **Clifford Data Regression** or CDR. Clifford circuits can be simulated efficiently in polynomial time using the stabilizer formalism. We generate thousands of random Clifford circuits, get their exact ideal values, then simulate them with noise.

> We also apply **Pauli Twirling** - inserting random Pauli gates around our CNOTs. This converts coherent errors into stochastic errors, which neural networks can learn more easily."

**Key Stats:**
- 7,010 total training samples
- 60% Clifford, 20% QAOA, 20% VQE circuits
- Variable noise scales: 0.5x to 2.5x

### 2.3 QEM-Former Architecture (1.5 min)
*[Screen: Architecture diagram]*

**SPEAKER:**
> "The second innovation is our architecture. We represent quantum circuits as **Directed Acyclic Graphs** - nodes are gates, edges are qubit wires connecting operations.

> The QEM-Former has five stages:
> 1. **Node Embedding**: Each gate type (H, CNOT, RZ) gets a learned 64-dimensional vector
> 2. **TransformerConv Layers**: Two graph attention layers capture local circuit topology
> 3. **Global Pooling**: Aggregate all node features into a single graph embedding
> 4. **Context Fusion**: Inject noise information - the noisy measurement, ZZ correlation, qubit count, circuit depth, and noise scale
> 5. **Regression Head**: MLP outputs the predicted ideal expectation value

> This architecture captures both the circuit structure AND the noise context - something traditional methods can't do."

---

## PART 3: RESULTS (2 minutes)

### 3.1 Benchmark Overview (45 sec)
*[Screen: Results table]*

**SPEAKER:**
> "We benchmarked against raw noisy execution and classical Zero-Noise Extrapolation across three circuit families.

> On Variational Ansatz circuits, we achieved an **80% win rate** - meaning 8 out of 10 predictions are closer to ideal than the raw noisy measurement. That's a **31.9% error reduction**.

> On Random Clifford circuits, we achieved 66.7% win rate with 31.2% error reduction.

> These results demonstrate our model generalizes from training data to new, unseen circuits."

**Key Table:**
| Circuit Type | Win Rate | Error Reduction |
|--------------|----------|-----------------|
| Variational  | **80%**  | **31.9%**       |
| Clifford     | 66.7%    | 31.2%           |
| QAOA         | 15%      | -115%           |

### 3.2 Honest Failure Analysis (45 sec)
*[Screen: QAOA failure explanation]*

**SPEAKER:**
> "We want to be transparent about our failure case. QAOA circuits performed poorly at only 15% win rate.

> The root cause: QAOA circuits have ideal expectation values very close to zero. Our model, trained mostly on Clifford circuits where values are ±1 or 0, overcorrects toward ±0.05, which increases error on near-zero targets.

> This is fixable by increasing QAOA training data and adding circuit-type awareness."

### 3.3 Architecture Evolution (30 sec)
*[Screen: Architecture comparison chart]*

**SPEAKER:**
> "We iteratively tested four architectures. SVR and LSTM achieved 0.03 MSE. GCN improved to 0.02. Our final QEM-Former achieved **0.009 MSE** - 3.3x better than the baseline.

> This validates that capturing circuit topology as a graph is essential for quantum error mitigation."

---

## PART 4: DEMO (2 minutes)
*[Screen: Live dashboard]*

**SPEAKER:**
> "Let me show you our interactive Streamlit dashboard.

> Here you can see the benchmark results, experiment with different circuit parameters, and visualize how our model corrects noisy measurements in real-time.

> [Walk through: Select circuit type, adjust noise, show prediction vs actual]

> The dashboard lets you explore how QEM-Former performs across different scenarios and noise levels."

---

## PART 5: IMPACT & CONCLUSION (1 minute)

### 5.1 Scalability & Business Value
*[Screen: Impact slide]*

**SPEAKER:**
> "This work has real impact for quantum computing in Africa and beyond:

> **Scalability**: Our graph representation is naturally extensible to larger circuits. With techniques like circuit knitting and transfer learning, this can scale to 100+ qubits.

> **Industry Relevance**: Every quantum computing company - IBM, Google, Amazon - needs error mitigation. Our approach is complementary to hardware improvements and requires no additional quantum resources.

> **African Development**: Quantum technology is a frontier where Africa can lead, not follow. This work demonstrates that African teams can contribute cutting-edge research."

### 5.2 Conclusion
*[Screen: Summary slide]*

**SPEAKER:**
> "To summarize: We built a complete, documented, reproducible pipeline for data-driven quantum error mitigation. Our QEM-Former achieves 31.9% error reduction on variational circuits with 80% win rate. We're transparent about limitations - QAOA needs more work. And we're excited about the potential for continued development.

> Thank you to the African Quantum Consortium for this opportunity. We're Team 15 - Nakahosa, Favour, and Abdulmalek."

---

## SPEAKER NOTES & TIPS

### Timing Breakdown
| Section | Duration |
|---------|----------|
| Introduction | 1:00 |
| Methodology | 3:00 |
| Results | 2:00 |
| Demo | 2:00 |
| Impact & Conclusion | 1:00 |
| **Total** | **9:00** |

### Key Phrases to Emphasize
- "31.9% error reduction"
- "80% win rate"
- "Graph Transformer captures circuit topology"
- "Clifford Data Regression overcomes supervision bottleneck"
- "Transparent about limitations"

### Visual Cues
- When showing architecture: point to each component as you describe it
- When showing results: highlight the green success bars and red failure
- During demo: click slowly enough for audience to follow

### Backup Answers for Q&A
- **Why simulated noise?** Reproducibility and exact ground truth access
- **Why not hardware?** Live calibration varies; simulated gives controlled experiments
- **How to fix QAOA?** Increase training proportion from 8% to 30%+
- **Scalability?** Transfer learning from small circuits, circuit knitting
