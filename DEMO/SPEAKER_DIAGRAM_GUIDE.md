# Speaker-Diagram Assignment Guide
## Team 15: The De-Noisers

---

## Nakahosa's Slides

### Slide 1 - Title (Introduction) — ~1:00
**File:** `slide_01_title_v2_*.png`

**Full Script:**
> "Hello everyone, and welcome! We are Team 15 - The De-Noisers - from the Hack the Horizon Hackathon, proudly hosted by the African Quantum Consortium.
>
> I'm Nakahosa, and I'm here with my teammates Favour and Abdulmalek. Today, we're excited to present our solution for one of quantum computing's most fundamental challenges: **quantum error mitigation**.
>
> Our project is called **QEM-Former** - a data-driven approach that uses a novel Graph Transformer architecture to learn how to correct noisy quantum measurements. 
>
> The problem is simple to state but hard to solve: NISQ devices - today's quantum computers - are extremely noisy. Every measurement we take is corrupted by thermal relaxation, dephasing, and readout errors. Our goal is to learn a correction function that recovers the ideal quantum expectation values from these noisy measurements.
>
> Let me walk you through our approach and results."

---

### Slide 2 - The Challenge (Problem Equation) — ~0:30
**File:** `slide_02_problem_v2_*.png`

**Full Script:**
> "Let me explain the core problem mathematically.
>
> When we measure an observable O on an ideal quantum state ρ, we get the true expectation value - that's the left side of this equation. But in reality, every quantum device applies a noise channel N to our state before measurement. This gives us a corrupted result - shown on the right.
>
> **Our challenge is learning to map from noisy measurements back to ideal values.** This is a regression problem, and we solve it using machine learning trained on carefully constructed datasets.
>
> The key insight is that if we can generate enough (noisy, ideal) training pairs, a neural network can learn to correct these errors."

---

### Slide 5 - Benchmark Results — ~0:45
**File:** `slide_05_results_v2_*.png`

**Full Script:**
> "Now let's look at our results. We benchmarked QEM-Former against two baselines: raw noisy execution and classical Zero-Noise Extrapolation, or ZNE.
>
> **On Variational Ansatz circuits** - the kind used in VQE chemistry simulations - we achieved an **80% win rate**. That means 8 out of 10 predictions were closer to the ideal value than the raw noisy measurement. This translated to a **31.9% error reduction**.
>
> **On Random Clifford circuits**, we achieved 66.7% win rate with 31.2% error reduction. These are slightly easier since our training data was primarily Clifford-based.
>
> These results demonstrate that our model **generalizes from training data to new, unseen circuits** - it's not just memorizing, it's learning the underlying error correction patterns.
>
> The table shows the full breakdown - you can see the stark contrast between our successes and our one failure case, which I'll discuss next."

---

### Slide 6 - QAOA Failure Analysis — ~0:45
**File:** `slide_06_qaoa_failure_v2_*.png`

**Full Script:**
> "We believe in being **transparent about our limitations**, so let me explain our failure case with QAOA circuits.
>
> QAOA performed poorly - only **15% win rate** and negative 115% error change, meaning we actually made things worse.
>
> **Why did this happen?** The root cause is data distribution mismatch. QAOA circuits have ideal expectation values that cluster around zero - very close to 0. But our training data was dominated by Clifford circuits, where ideal values are typically ±1 or exactly 0.
>
> So when our model sees a QAOA circuit, it overcorrects toward ±0.05 instead of leaving the near-zero value alone. It's essentially applying Clifford-style corrections to a QAOA circuit.
>
> **The fix is straightforward**: increase QAOA representation in training data from 8% to 30% or more. This is a **data distribution problem, not an architecture problem** - the model can learn QAOA patterns if given enough examples.
>
> We chose to present this honestly because it shows where future work should focus."

---

### Slide 9 - Impact & Business Value — ~0:30
**File:** `slide_09_impact_v2_*.png`

**Full Script:**
> "This work has real impact beyond the hackathon.
>
> **Scalability**: Our graph representation is naturally extensible. With techniques like circuit knitting and transfer learning, this approach can scale to 100+ qubit systems.
>
> **Industry Relevance**: Every major quantum computing company - IBM, Google, Amazon - needs error mitigation. Our approach complements hardware improvements and requires no additional quantum resources. It's purely classical post-processing.
>
> **African Development**: Quantum technology is a frontier where Africa can lead, not follow. This project demonstrates that African teams can contribute cutting-edge research to the global quantum ecosystem.
>
> We see this as a starting point for continued development and potential commercialization."

---

## Favour's Slides

### Slide 3 - Data Pipeline (CDR + Pauli Twirling) — ~1:00
**File:** `slide_03_data_pipeline_v2_*.png`

**Full Script:**
> "The first major innovation in our approach is our **data generation pipeline**.
>
> Training a neural network for error mitigation requires pairs of noisy and ideal measurements. But here's the problem: computing ideal values for arbitrary large quantum circuits is **exponentially expensive** - it requires full statevector simulation.
>
> We solve this using **Clifford Data Regression**, or CDR. Clifford circuits are special - they can be simulated efficiently in **polynomial time** using the stabilizer formalism, thanks to the Gottesman-Knill theorem. So we generate thousands of random Clifford circuits, compute their exact ideal values efficiently, then simulate them with realistic noise.
>
> We also apply **Pauli Twirling** - this technique inserts random Pauli gates around our CNOT operations. The effect is converting coherent errors into stochastic errors. Why does this matter? Neural networks learn stochastic patterns much more easily than coherent ones.
>
> Our final dataset has **7,010 training samples**: 60% Clifford, 20% QAOA, and 20% VQE-style circuits. We also vary noise scales from 0.5x to 2.5x to teach the model to handle different noise environments."

---

### Slide 3b - Deep Dive: CDR & Pauli Twirling Details (OPTIONAL) — ~0:30
**File:** `cdr_pauli_twirling_1768503241552.png`

**Full Script:**
> "Let me show you the details in this diagram.
>
> On the left, you can see the CDR process: Clifford circuits flow through stabilizer simulation in O(poly(n)) time to produce exact ground truth values. The Gottesman-Knill theorem is the key enabler here.
>
> On the right, Pauli Twirling is illustrated. We insert random Pauli gates Pc and Pt around each CNOT, then their conjugates afterward. This randomization converts amplitude and phase noise - which is coherent - into probabilistic bit and phase flips - which are stochastic.
>
> The bottom line shows why this matters: **twirling simplifies the error landscape**, making it more learnable for our neural network."

**Gesture:** Point to each section as you explain.

---

### Slide 4 - QEM-Former Architecture — ~1:30
**File:** `slide_04_architecture_v2_*.png`

**Full Script:**
> "The second major innovation is our **QEM-Former architecture**.
>
> We represent quantum circuits as **Directed Acyclic Graphs** - nodes are gates, edges are qubit wires connecting operations in sequence. This captures the circuit topology in a way that standard neural networks cannot.
>
> The architecture has **five key stages**:
>
> **First, Node Embedding.** Each gate type - Hadamard, CNOT, RZ rotations - gets a learned 64-dimensional vector representation. The model learns what each gate 'means' for error correction.
>
> **Second, TransformerConv Layers.** We use two graph attention layers that capture local circuit topology. These layers let each gate 'attend' to its neighbors, learning how errors propagate through the circuit.
>
> **Third, Global Pooling.** We aggregate all node features into a single graph-level embedding using mean pooling. This gives us a fixed-size representation regardless of circuit size.
>
> **Fourth, Context Fusion.** Here we inject crucial noise information: the noisy measurement value, ZZ correlation, qubit count, circuit depth, and noise scale. This context tells the model about the specific noise environment.
>
> **Finally, the Regression Head** - a 3-layer MLP that outputs the predicted ideal expectation value.
>
> The key insight is that this architecture captures **both circuit structure AND noise context** - something traditional physics-based methods cannot do."

---

### Slide 7 - Architecture Evolution — ~0:30
**File:** `slide_07_arch_comp_v2_*.png`

**Full Script:**
> "We didn't arrive at QEM-Former by accident - we iteratively tested and improved through four architectures.
>
> **Support Vector Regression** and **LSTM** both achieved 0.03 MSE. These are strong baselines but they don't capture circuit structure.
>
> **Graph Convolutional Networks** improved to 0.02 MSE - the graph structure helped, but standard convolutions have limited receptive fields.
>
> Our final **QEM-Former** achieved **0.009 MSE** - that's **3.3x better than baseline**!
>
> This validates our core hypothesis: capturing circuit topology as a graph, combined with transformer-style attention, is essential for quantum error mitigation."

---

### Slide 8 - Interactive Demo — ~2:00
**File:** `slide_08_dashboard_v2_*.png`

**Full Script:**
> "Now let me show you our interactive Streamlit dashboard - and this is a live demonstration.
>
> *[Switch to live dashboard]*
>
> Here in the sidebar, you can select different circuit types: Random Clifford, QAOA MaxCut, or Variational Ansatz. You can also adjust the number of qubits and the noise scale factor.
>
> Let me run a live prediction. I'll select Variational Ansatz with 5 qubits and noise scale 1.0...
>
> *[Click Run Mitigation]*
>
> Watch the results appear: you can see the **True Value** from ideal simulation, the **ZNE Baseline**, and our **QEM-Former prediction**. The improvement ratio shows how much better our model performed.
>
> We also have additional visualization tabs: Error by Qubit shows per-qubit error distribution, Connectivity shows the qubit interaction graph, and Circuit Viewer displays the actual quantum circuit.
>
> You can also use our **Presentation Mode** tab which walks through each step of our methodology with live simulations - perfect for demonstrating to judges or collaborators.
>
> Feel free to experiment with different parameters - the model runs in real-time!"

---

## Both Speakers

### Slide 10 - Summary & Thank You — ~0:30
**File:** `slide_10_summary_v2_*.png`

**Full Script (Nakahosa):**
> "To summarize what we've built:
> - A **complete, documented, reproducible pipeline** for data-driven quantum error mitigation
> - **31.9% error reduction** on variational circuits with **80% win rate**
> - We're **transparent about limitations** - QAOA needs more training data
> - And we're excited about the potential for scaling and continued development."

**Full Script (Favour):**
> "We want to thank the **African Quantum Consortium** for this incredible opportunity to contribute to quantum computing research.
>
> We are **Team 15 - The De-Noisers**: Nakahosa, Favour, and Abdulmalek.
>
> Thank you for your time, and we're happy to answer any questions!"

---

## Quick Reference Table

| Slide | Title | Speaker | Duration |
|-------|-------|---------|----------|
| 1 | Title | Nakahosa | ~1:00 |
| 2 | Challenge | Nakahosa | ~0:30 |
| 3 | Data Pipeline | Favour | ~1:00 |
| 3b | CDR & Pauli Deep Dive (Optional) | Favour | ~0:30 |
| 4 | Architecture | Favour | ~1:30 |
| 5 | Results | Nakahosa | ~0:45 |
| 6 | QAOA Failure | Nakahosa | ~0:45 |
| 7 | Architecture Evolution | Favour | ~0:30 |
| 8 | Dashboard Demo | Favour | ~2:00 |
| 9 | Impact | Nakahosa | ~0:30 |
| 10 | Summary | Both | ~0:30 |

**Total: ~9 minutes**

---

## Presentation Tips

### For Nakahosa:
- Speak slowly and clearly on the mathematical equations
- Emphasize the "80% win rate" and "31.9% error reduction" numbers
- On the QAOA slide, use a confident tone - honesty is a strength

### For Favour:
- Point to diagram elements as you explain them
- For the architecture slide, pause between each of the 5 stages
- During the live demo, narrate what you're clicking
- Have a backup plan if the demo doesn't work (pre-recorded video)

### General:
- Practice handoffs between speakers
- Make eye contact with the camera
- Keep energy high throughout
