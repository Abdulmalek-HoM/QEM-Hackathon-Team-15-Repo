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

print(f"Simulator n_qubits: {sim.configuration().n_qubits}")
if getattr(sim.configuration(), 'coupling_map', None):
    print(f"Simulator coupling_map size: {len(sim.configuration().coupling_map)}")
else:
    print("No coupling_map in sim config")

try:
    qc_t = transpile(qc, sim)
    print("Transpile successful")
except Exception as e:
    print(f"Transpile error: {e}")
