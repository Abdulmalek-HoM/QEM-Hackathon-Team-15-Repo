# Team 15: Data-Driven Quantum Error Mitigation (QEM)

## Overview
This repository contains the research and codebase for Team 15's submission to the **Aqora "Data-Driven Quantum Error Mitigation" Competition**. Our goal is to develop a robust, data-driven pipeline to mitigate noise in Intermediate-Scale Quantum (NISQ) devices.

We move beyond traditional error mitigation techniques (like Zero-Noise Extrapolation or ZNE) by implementing advanced **Deep Learning models**, specifically **Graph Transformers (QEMFormer)** and **LSTM-based architectures**, to learn and correct complex noise patterns.

## Project Structure

The project is structured into educational and practical modules, guiding the user from basic quantum operations to advanced AI-based error mitigation.

### Educational & Dev Modules
*   **`Module_1_Qubit_Operations.ipynb`**: Introduction to basic qubit manipulations.
*   **`Module_2_Circuits_Connectivity.ipynb`**: Understanding circuit topology and connectivity constraints.
*   **`Module_3_Large_Scale.ipynb`**: Strategies for handling larger quantum circuits.
*   **`Module_4_Modeling_Errors.ipynb`**: simulating and analyzing quantum noise channels.
*   **`Module_5_ZNE.ipynb`**: Implementation of Zero-Noise Extrapolation (ZNE) using Mitiq.
*   **`Module_6_AI_Models.ipynb`**: Developing and training AI models (LSTMs, Transformers) for QEM.
*   **`Module_7_Deployment.ipynb`**: Deployment pipeline for the trained QEM models.

### Core Source Code
*   **`data_gen_advanced.py`**: A robust data generation pipeline utilizing **Clifford Data Regression (CDR)** and **Pauli Twirling**. Generates dataset graphs for training.
*   **`models/qem_former.py`**: Implementation of the **QEM-Former**, a Graph Transformer model with learned embeddings for gates and noise data.
*   **`train_qem.py`**: Script to train the QEM models using the generated datasets.
*   **`benchmark_suite.py`**: A benchmarking suite to evaluate the model's performance (Improvement Ratio) against baselines like ZNE.
*   **`dashboard.py`**: A Streamlit dashboard for visualizing results and interacting with the models.

## Installation

1.  **Clone the repository** (if applicable).
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Generation
To generate a training dataset using Clifford Data Regression and Pauli Twirling:
```bash
python data_gen_advanced.py
```
This will create dataset files in the `dataset/` directory.

### 2. Training
To train the QEM-Former model:
```bash
python train_qem.py
```
Trained models will be saved in the `models/` directory or as `.pth` files in the root.

### 3. Benchmarking
To evaluate the model's performance:
```bash
python benchmark_suite.py
```

### 4. Interactive Dashboard
To explore the models and results interactively:
```bash
streamlit run dashboard.py
```

## Methodology

Our approach follows a three-phase strategy:

1.  **Data Generation**: Overcoming the supervision bottleneck using **Clifford Data Regression (CDR)** to generate training pairs ($x_{noisy}$, $y_{ideal}$) from efficiently simulable near-Clifford circuits. We augment this with **Variable-Noise Training** and **Pauli Twirling**.
2.  **Model Architecture**: We utilize **Graph Transformers (QEMFormer)** to capture the non-local entanglement and structure of quantum circuits. The model treats the circuit as a Directed Acyclic Graph (DAG), learning to map noisy states to improved expectation values.
3.  **Validation**: Rigorous benchmarking using the **Improvement Ratio (IR)** metric to quantify gain over raw hardware execution and standard ZNE.

## Authors
**Aqora Competition Strategy Team (Team 15)**
