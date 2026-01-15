# Technical Code Demo Script
## Live Code Walkthrough for Video Presentation

This guide shows **exactly what screens to show** and **what to say** while demonstrating your technical implementation.

---

## SETUP BEFORE RECORDING

```bash
# Terminal 1: Ready to run commands
cd "/Users/abdulmalekbaitulmal/Downloads/Desktop/Quanta Related/AQC/AQC Hack The Horizon/QEM Codebase"

# Editor: Have these files open in tabs
# 1. models/qem_former.py
# 2. data_gen_advanced.py  
# 3. train_qem.py
# 4. benchmark_suite.py
```

---

## PART 1: DATA GENERATION CODE (1.5 min)

### Screen: `data_gen_advanced.py` in editor

**SHOW: Lines 117-220 (Pauli Twirling)**

**SAY:**
> "Let me show you our Pauli Twirling implementation. This is the key to making noise learnable.

> Here at line 117 we have `apply_pauli_twirling`. For each CNOT gate, we insert random Pauli gates before and after. See the lookup table at line 198 - this ensures we preserve the logical CNOT operation while randomizing the noise.

> The effect: coherent errors become stochastic. Neural networks learn stochastic patterns much better."

---

**SHOW: Lines 267-360 (Main data generation)**

**SAY:**
> "Here's the main data generation. Line 295 - we use a stabilizer simulator for Clifford circuits. This is O(poly(n)) instead of O(2^n).

> Line 329 - we run the twirled circuit through a noisy simulator.

> Line 335 - we build global features: [z0_noisy, zz_correlation, qubit_count, depth, noise_scale]. This is the noise context our model learns from.

> Line 340 - our target is z0_ideal from the exact simulation."

---

## PART 2: MODEL ARCHITECTURE CODE (1.5 min)

### Screen: `models/qem_former.py` in editor

**SHOW: Lines 8-18 (Class docstring)**

**SAY:**
> "This is QEMFormer - our Graph Transformer for quantum error mitigation.

> The architecture has 5 stages: Node Embedding, Local GNN, Global Transformer, Readout, and Context Fusion."

---

**SHOW: Lines 28-74 (Model init)**

**SAY:**
> "Line 29 - we embed each gate type into a 64-dimensional vector. H gets one embedding, CNOT gets another.

> Line 30 - rotation parameters also get encoded.

> Lines 34-38 - TransformerConv layers. These are attention-based message passing on the circuit graph. Each gate attends to connected gates.

> Line 66 - this is crucial. We encode the noise context - the noisy measurement, qubit count, depth, noise scale. This tells the model ABOUT the noise.

> Lines 68-74 - the final regressor combines graph embedding plus context embedding to predict the ideal value."

---

**SHOW: Lines 76-113 (Forward pass)**

**SAY:**
> "The forward pass shows the data flow.

> Line 87 - node features are gate embeddings plus parameter encodings.

> Lines 90-92 - we run through the TransformerConv layers. This captures local circuit topology.

> Line 105 - global mean pooling aggregates all gate features into one graph embedding.

> Line 108 - context fusion. We concatenate the graph embedding with the noise context.

> Line 112 - output is the predicted ideal expectation value."

---

## PART 3: TRAINING PIPELINE (1 min)

### Screen: Terminal

**RUN:**
```bash
# Show dataset files
ls -la dataset/
```

**SAY:**
> "We have 10 chunks of Clifford data plus a mixed dataset. Total 7,010 training samples."

---

**SHOW: `train_qem.py` lines 86-145**

**SAY:**
> "Line 90 - we train for 100 epochs.

> Lines 94-106 - standard PyTorch training loop. We reshape global attributes to match batch size, forward pass through the model, compute MSE loss, backprop.

> Line 124 - we use ReduceLROnPlateau scheduler. If loss plateaus, learning rate drops.

> Line 139 - we save the best model based on validation loss."

---

### Screen: Show `training_curves.png`

**SAY:**
> "Here's the training result. Loss dropped from 0.24 to 0.009. Best model was at epoch 22. The learning rate decayed 5 times as you can see in the right plot."

---

## PART 4: BENCHMARK EXECUTION (1 min)

### Screen: Terminal

**RUN:**
```bash
python benchmark_suite.py
```

**SAY:**
> "Let's run the benchmark suite. This tests on three circuit families: Random Clifford, QAOA, and Variational.

> Watch the output...

> [As it runs] We're generating circuits, computing ground truth via statevector, running noisy simulation, and comparing our model's prediction.

> [After completion] See the results: 80% win rate on Variational, 66.7% on Clifford, and 15% on QAOA - our known failure case."

---

**IF TIME IS SHORT:** Just show the pre-computed results:

```bash
cat benchmark_results.json | python -m json.tool
```

**SAY:**
> "Here are our saved benchmark results. Mean improvement ratio of 1.44x on Variational - that's 31.9% error reduction."

---

## PART 5: LIVE DASHBOARD (1 min)

### Screen: Browser with Streamlit

**ALREADY RUNNING:** The dashboard should be running from earlier.

**WALK THROUGH:**
1. Show benchmark results tab
2. Adjust parameters (qubits, noise)
3. Show live prediction

**SAY:**
> "The dashboard brings everything together. Here we can explore results, adjust noise parameters, and see the model make predictions in real-time."

---

## KEY CODE SNIPPETS TO HIGHLIGHT

### Pauli Twirling (STAR moment)
```python
# Lines 156-163
paulis = ['id', 'x', 'y', 'z']
idx_c = np.random.randint(4)
idx_t = np.random.randint(4)
# Insert before CNOT
if pc_gate != 'id': getattr(twirled_qc, pc_gate)(qubits[0])
if pt_gate != 'id': getattr(twirled_qc, pt_gate)(qubits[1])
twirled_qc.cx(qubits[0], qubits[1])
# Insert correction after
```

### Graph Construction (STAR moment)
```python
# Lines 92-104
for node in dag.topological_op_nodes():
    for q in node.qargs:
        prev_idx = last_node_on_qubit[q]
        if prev_idx != -1:
            edge_index[0].append(prev_idx)
            edge_index[1].append(current_idx)
        last_node_on_qubit[q] = current_idx
```

### Context Fusion (STAR moment)
```python
# Lines 105-112
graph_embed = self.pool(h, batch)
context_embed = F.relu(self.context_encoder(global_attr))
combined = torch.cat([graph_embed, context_embed], dim=1)
out = self.regressor(combined)
```

---

## TIMING GUIDE

| Section | Duration | What's on Screen |
|---------|----------|------------------|
| Data Generation | 1:30 | `data_gen_advanced.py` |
| Model Architecture | 1:30 | `models/qem_former.py` |
| Training Pipeline | 1:00 | `train_qem.py` + terminal |
| Benchmark Execution | 1:00 | Terminal + `benchmark_results.json` |
| Dashboard Demo | 1:00 | Browser |
| **Total Technical** | **6:00** | |

---

## EMERGENCY FALLBACKS

### If code scrolling is awkward:
Use SLIDES_CONTENT.md code snippets as pre-formatted screenshots instead.

### If benchmark takes too long:
Just `cat benchmark_results.json` and explain the pre-computed results.

### If terminal has errors:
Say: "Let me show you the saved outputs instead" and switch to showing output files.

---

## POST-DEMO CLOSER

**SAY:**
> "All of this code is open source and available in our GitHub repository. You can reproduce everything we've shown today.

> The key innovations: CDR for training data, Pauli Twirling for learnable noise, and Graph Transformers for circuit topology. Together, they achieve 31.9% error reduction on variational circuits."
