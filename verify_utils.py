import utils
from qiskit import QuantumCircuit

print("Testing Utils...")
try:
    nm = utils.build_noise_model(scale=2.0)
    print("✅ Noise Model Built.")
except Exception as e:
    print(f"❌ Noise Model Failed: {e}")

try:
    qc, instr = utils.create_random_clifford_circuit(2, 5)
    print(f"✅ Circuit Created. Depth: {qc.depth()}")
    print(f"Instructions: {instr}")
except Exception as e:
    print(f"❌ Circuit Gen Failed: {e}")

try:
    tok = utils.CircuitTokenizer(max_length=10)
    seq = tok.tokenize(instr)
    print(f"✅ Tokens: {seq}")
    assert len(seq) == 10
    print("✅ Padding Verification utilizing max_length Passed.")
except Exception as e:
    print(f"❌ Tokenizer Failed: {e}")

print("Utils Verification Complete.")
