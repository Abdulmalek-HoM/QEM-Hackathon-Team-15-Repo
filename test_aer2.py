import warnings
warnings.filterwarnings('ignore')
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import utils

qc = QuantumCircuit(48)
qc.h(0)
qc.cx(0, 1)

nm = utils.build_noise_model(scale=1.0)
sim = AerSimulator(noise_model=nm)

try:
    qc_t = transpile(qc, basis_gates=sim.operation_names)
    print("Transpile successful with basis_gates")
except Exception as e:
    print(f"Transpile error: {e}")
