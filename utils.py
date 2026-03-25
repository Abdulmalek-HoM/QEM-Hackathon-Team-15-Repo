import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError

def build_noise_model(scale=1.0):
    """
    Standard noise model for QEM Hackathon.
    Includes T1/T2 thermal relaxation and Readout error.
    Args:
        scale (float): Noise scaling factor (1.0 = baseline). 
                       Higher scale = worse noise (shorter T1/T2, higher readout error).
    """
    noise_model = NoiseModel()
    
    # Scale constants
    # Worse noise means SHORTER coherence time -> Divide by scale
    t1 = 50e-6 / scale
    t2 = 70e-6 / scale
    
    # Worse noise means HIGHER readout error -> Multiply by scale
    p_ro = min(0.5, 0.05 * scale)
    
    # Error probabilities
    # 1-qubit gate error (short duration)
    error_1q = thermal_relaxation_error(t1, t2, 50e-9)
    # 2-qubit gate error (long duration)
    error_2q_single = thermal_relaxation_error(t1, t2, 400e-9)
    error_2q = error_2q_single.expand(error_2q_single)
    
    # Add errors
    noise_model.add_all_qubit_quantum_error(error_1q, ['x', 'h', 'id', 'z', 's', 'sdg', 'y'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    # Readout error
    probs = [[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]
    noise_model.add_all_qubit_readout_error(ReadoutError(probs))
    
    return noise_model

def create_random_clifford_circuit(num_qubits, depth):
    """
    Generates a random Clifford circuit and returns both the object and the instruction list.
    Returns: (QuantumCircuit, list[str])
    """
    qc = QuantumCircuit(num_qubits)
    gates_1q = ['h', 's', 'x', 'y', 'z', 'id']
    instructions = [] 
    
    for _ in range(depth):
        q = random.randint(0, num_qubits - 1)
        if num_qubits > 1 and random.random() > 0.5:
             # Pick random target different from control
             target = random.randint(0, num_qubits - 1)
             while target == q:
                 target = random.randint(0, num_qubits - 1)
             
             qc.cx(q, target)
             instructions.append(f"cx {q} {target}")
        else:
             g = random.choice(gates_1q)
             getattr(qc, g)(q)
             instructions.append(f"{g} {q}")
             
    return qc, instructions

def create_variational_circuit(num_qubits, depth):
    """
    Creates a Hardware-Efficient Ansatz (common in Variational Algorithms).
    Layers of RY rotations and CNOT entanglers.
    Returns: (QuantumCircuit, list[str])
    """
    qc = QuantumCircuit(num_qubits)
    instructions = []
    
    for _ in range(depth):
        # Rotation Layer
        for q in range(num_qubits):
            # Random angle for benchmark variety
            theta = np.random.uniform(0, 2*np.pi)
            qc.ry(theta, q)
            # Approximate instruction for tokenizer (binning angle not needed for simple tokenization)
            instructions.append(f"ry {q}")
            
        # Entanglement Layer
        if num_qubits > 1:
            for q in range(num_qubits - 1):
                qc.cx(q, q+1)
                instructions.append(f"cx {q} {q+1}")
                
    return qc, instructions

def create_qaoa_circuit(num_qubits, p=1):
    """
    Creates a dummy QAOA-like structure (Cost + Mixer layers).
    Returns: (QuantumCircuit, list[str])
    """
    qc = QuantumCircuit(num_qubits)
    instructions = []
    
    # Initial Superposition
    for q in range(num_qubits):
        qc.h(q)
        instructions.append(f"h {q}")
        
    for _ in range(p):
        # Cost Hamiltonian (ZZ interactions)
        if num_qubits > 1:
            for q in range(num_qubits - 1):
                gamma = np.random.uniform(0, 2*np.pi)
                qc.rzz(gamma, q, q+1)
                # Tokenizer treats rzz as generic operation if not in vocab, 
                # or we can decompose it. For simplicity, we stick to standard tokens if possible.
                # But our tokenizer currently supports ['h', 's', 'x', 'y', 'z', 'id', 'cx'].
                # So we decompose RZZ roughly into CNOT-RZ-CNOT for tokenization purposes or add to vocab.
                # Let's append standard gates that represent the complexity.
                instructions.append(f"cx {q} {q+1}")
                instructions.append(f"rz {q+1}")
                instructions.append(f"cx {q} {q+1}")
        
        # Mixer Hamiltonia (X rotations)
        for q in range(num_qubits):
            beta = np.random.uniform(0, 2*np.pi)
            qc.rx(beta, q)
            instructions.append(f"rx {q}")
            
    return qc, instructions


class CircuitTokenizer:
    def __init__(self, max_length=50):
        self.vocab = {
            "<PAD>": 0,
            "h": 1, "s": 2, "x": 3, "y": 4, "z": 5, "id": 6, "cx": 7
        }
        self.max_length = max_length

    def tokenize(self, instruction_list):
        """
        Converts list of instruction strings (e.g. 'h 0') to list of integers.
        Ignores qubit indices for this simple embedding (treating 'h 0' same as 'h 1').
        """
        token_seq = []
        for instr in instruction_list:
            parts = instr.split()
            gate_name = parts[0]
            if gate_name in self.vocab:
                token_seq.append(self.vocab[gate_name])
        
        # Padding / Truncating
        if len(token_seq) < self.max_length:
            token_seq += [self.vocab["<PAD>"]] * (self.max_length - len(token_seq))
        else:
            token_seq = token_seq[:self.max_length]
            
        return token_seq

import psutil
import time

class LiveHardwareProfiler:
    """Helper class to display real-time hardware usage in Streamlit."""
    def __init__(self, container):
        self.container = container
        self.start_time = time.time()
        
    @staticmethod
    def display_compute_estimate(container, n_qubits, depth, n_samples=1):
        """Displays a pre-execution heuristic estimate of RAM and processing time."""
        import psutil
        import streamlit as st
        
        # Statevector memory: 2^n complex numbers (16 bytes each)
        # We need at least 2-3 copies for simulation overhead
        bytes_per_state = (2 ** n_qubits) * 16
        est_ram_gb = (bytes_per_state * 3) / (1024 ** 3)
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        
        # Rough time heuristic: (depth * 2^n) ops per shot
        # A modern CPU might do 1e9 ops per sec
        ops_per_circuit = depth * (2 ** n_qubits)
        est_time_sec = (ops_per_circuit * n_samples) / 1e8  # Heavily padded heuristic
        
        with container.expander("⚖️ Pre-Execution Compute Estimator", expanded=(est_ram_gb > available_ram_gb * 0.5)):
            st.markdown(f"**Target Qubits:** `{n_qubits}` | **Samples:** `{n_samples}`")
            col1, col2 = st.columns(2)
            
            if est_ram_gb < 0.01:
                col1.metric("Estimated RAM Overhead", "< 10 MB")
            else:
                col1.metric("Estimated RAM Overhead", f"{est_ram_gb:.2f} GB", 
                            delta="Danger!" if est_ram_gb > available_ram_gb else "Safe", 
                            delta_color="inverse" if est_ram_gb > available_ram_gb else "normal")
                            
            if est_time_sec < 1:
                col2.metric("Estimated Processing Time", "< 1 sec")
            elif est_time_sec < 60:
                col2.metric("Estimated Processing Time", f"{est_time_sec:.1f} secs")
            else:
                col2.metric("Estimated Processing Time", f"{est_time_sec/60:.1f} mins")
                
            if est_ram_gb > available_ram_gb:
                st.error("⚠️ **CRITICAL WARNING:** This simulation mathematically exceeds your available physical memory! Your machine will likely freeze or crash if you execute this. Please reduce the number of qubits.")
                return False
            elif n_qubits >= 25:
                st.warning("⚠️ **MEMORY WARNING:** Simulating >25 qubits requires exponential RAM. Ensure you have closed other applications.")
            return True
        
    def update(self, step_name, active_files, circuits_processed=0, total_circuits=0, active_qc=None):
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024 ** 3)
            ram_total_gb = ram.total / (1024 ** 3)
            cores = psutil.cpu_count(logical=False)
            threads = psutil.cpu_count(logical=True)
            
            elapsed = time.time() - self.start_time
            
            flops_est = "N/A"
            if elapsed > 0 and circuits_processed > 0:
                # Very rough heuristic: Qiskit statevector takes O(2^n) ops. We just show theoretical circuit throughput.
                cps = circuits_processed / elapsed
                flops_est = f"{cps:.2f} circuits/sec"
            
            with self.container.container():
                import streamlit as st
                st.markdown("### 🖥️ Live Hardware Execution Profiler")
                st.caption("Active monitoring of compute resources allocated to QEM Pipeline.")
                
                st.markdown(f"**🟢 Current Step:** `{step_name}`")
                st.markdown(f"**📂 Active Codebase Scripts:** `{', '.join(active_files)}`")
                
                if active_qc is not None:
                    st.markdown(f"**⚛️ Quantum Circuit Topology:** `{active_qc.num_qubits} Qubits` | `Depth {active_qc.depth()}` | `{sum(dict(active_qc.count_ops()).values())} Gates`")
                
                c1, c2 = st.columns(2)
                c1.metric("CPU Allocation", f"{cpu_percent}%", f"{cores} Cores / {threads} Threads", delta_color="off")
                c2.metric("RAM Saturation", f"{ram.percent}%", f"{ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB", delta_color="inverse")
                
                if total_circuits > 0:
                    st.progress(circuits_processed / total_circuits, text=f"Processing Pipeline Throughput: {flops_est}")
                
                if st.button("⏹️ Force Stop Simulation", key=f"stop_btn_{time.time()}"):
                    st.session_state.stop_execution = True
                    st.warning("Halting execution! Please wait for the script to refresh...")
                    st.stop()
                
                st.markdown("---")
        except Exception as e:
            pass # Failsafe for Streamlit thread context
