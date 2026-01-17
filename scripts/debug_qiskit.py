import sys
print(f"Python executable: {sys.executable}")
try:
    import qiskit
    print(f"Qiskit file: {qiskit.__file__}")
    print(f"Qiskit path: {qiskit.__path__}")
    print(f"Dir(qiskit): {dir(qiskit)}")
    from qiskit import QuantumCircuit
    print("QuantumCircuit imported successfully")
except Exception as e:
    print(f"Error: {e}")
