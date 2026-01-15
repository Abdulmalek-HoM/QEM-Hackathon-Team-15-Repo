# Speaker-Diagram Assignment Guide
## Team 15: The De-Noisers

---

## Nakahosa's Slides

### Slide 1 - Title (Introduction)
**File:** `slide_01_title_*.png`
**Content:** Team intro, problem statement
> "Hello, we're Team 15 from the Hack the Horizon Hackathon. Today we're presenting our solution for data-driven quantum error mitigation using a novel Graph Transformer architecture we call QEM-Former..."

---

### Slide 2 - The Challenge (Problem Equation)
**File:** `slide_02_problem_*.png`
**Content:** Ideal vs noisy measurements equation
> "When we measure an observable O on an ideal quantum state ρ, we get the true expectation value. But with noise channel N, we get a corrupted result. The challenge is learning to map noisy → ideal."

---

### Slide 5 - Benchmark Results
**File:** `slide_05_results_*.png`
**Content:** Results table with win rates
> "We benchmarked against raw noisy execution and classical Zero-Noise Extrapolation across three circuit families. On Variational Ansatz circuits, we achieved an 80% win rate..."

---

### Slide 6 - QAOA Failure Analysis
**File:** `slide_06_qaoa_failure_*.png`
**Content:** Honest failure explanation
> "We want to be transparent about our failure case. QAOA circuits performed poorly at only 15% win rate. The root cause: QAOA circuits have ideal expectation values very close to zero..."

---

### Slide 9 - Impact & Business Value
**File:** `slide_09_impact_*.png`
**Content:** Scalability, Industry, African Development
> "This work has real impact for quantum computing in Africa and beyond: Scalability, Industry Relevance, African Development..."

---

## Favour's Slides

### Slide 3 - Data Pipeline (CDR + Pauli Twirling)
**File:** `slide_03_data_pipeline_*.png`
**Content:** Data generation methodology
> "The first innovation is our data generation. Training a neural network requires pairs of noisy and ideal measurements... We solve this using Clifford Data Regression or CDR..."

---

### Slide 3b - Deep Dive: CDR & Pauli Twirling Details (OPTIONAL)
**File:** `cdr_pauli_twirling_1768503241552.png`
**Content:** Detailed explanation of CDR and Pauli Twirling mechanisms
> "Let me show you in more detail how these techniques work. On the left, CDR uses the Gottesman-Knill theorem for stabilizer simulation in O(poly(n)) time to get exact ground truth. On the right, Pauli Twirling converts coherent errors into stochastic errors - neural networks learn stochastic patterns better."

**Key Points to Highlight:**
- CDR enables polynomial-time simulation via stabilizer formalism
- Pauli Twirling simplifies the error landscape
- Both techniques make the data more learnable for neural networks

---

### Slide 4 - QEM-Former Architecture
**File:** `slide_04_architecture_*.png`
**Content:** 5-stage architecture walkthrough
> "The second innovation is our architecture. We represent quantum circuits as Directed Acyclic Graphs - nodes are gates, edges are qubit wires. The QEM-Former has five stages..."

---

### Slide 7 - Architecture Evolution
**File:** `slide_07_architecture_comparison_*.png`
**Content:** Architecture comparison chart
> "We iteratively tested four architectures. SVR and LSTM achieved 0.03 MSE. GCN improved to 0.02. Our final QEM-Former achieved 0.009 MSE - 3.3x better than the baseline."

---

### Slide 8 - Interactive Demo
**File:** `slide_08_dashboard_*.png`
**Content:** Dashboard demonstration
> "Let me show you our interactive Streamlit dashboard. Here you can see the benchmark results, experiment with different circuit parameters..."

---

## Both Speakers

### Slide 10 - Summary & Thank You
**File:** `slide_10_summary_*.png`
**Content:** Conclusion and acknowledgments
> "To summarize: We built a complete, documented, reproducible pipeline for data-driven quantum error mitigation... Thank you to the African Quantum Consortium. We're Team 15 - Nakahosa, Favour, and Abdulmalek."

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
