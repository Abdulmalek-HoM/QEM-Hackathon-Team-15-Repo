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
