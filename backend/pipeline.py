import torch
import numpy as np
import sys
import os

# Add root directory to path to allow importing from utils and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.qem_lstm import QEM_LSTM
# from models.qem_former import QEMFormer # TODO: Enable when Transformer is ready
import utils
from qiskit_aer import AerSimulator
from qiskit import transpile
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

class HackathonPipeline:
    def __init__(self, lstm_path="qem_lstm.pth"):
        self.tokenizer = utils.CircuitTokenizer(max_length=60)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load AI Model (LSTM)
        try:
            # We need to know vocab size. 
            # Ideally this is stored or we re-instantiate tokenizer exactly as before.
            # For hackathon rig: default vocab size is usually fixed by CircuitTokenizer logic
            vocab_size = len(self.tokenizer.vocab) + 1
            self.model = QEM_LSTM(vocab_size).to(self.device)
            
            # Map location for safety if no GPU
            state_dict = torch.load(lstm_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✅ AI Model (LSTM) Loaded Successfully.")
        except FileNotFoundError:
            print(f"⚠️ Model file {lstm_path} not found. AI correction will be disabled.")
            self.model = None
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.model = None

    def run_zne(self, qc):
        """Runs Exponential ZNE."""
        scales = [1.0, 2.0, 3.0]
        results = []
        
        for s in scales:
            nm = utils.build_noise_model(scale=s)
            sim = AerSimulator(noise_model=nm)
            qc_t = transpile(qc, sim)
            counts = sim.run(qc_t, shots=1000).result().get_counts()
            shots = sum(counts.values())
            # Calculate expectation value Z
            # (P00+P11) - (P01+P10) for 2 qubits target
            # TODO: Generalize for N qubits if needed, currently hardcoded for 2 from Module 7
            val = (counts.get('00', 0)+counts.get('11', 0) - counts.get('01', 0)-counts.get('10', 0))/shots
            results.append(val)
            
        # Extrapolate
        def exp_decay(x, a, b, c): return a * np.exp(-b * x) + c
        try:
            popt, _ = curve_fit(exp_decay, scales, results, p0=[1, 0.1, 0], maxfev=2000)
            base_est = exp_decay(0, *popt)
        except:
            lr = LinearRegression()
            lr.fit(np.array(scales).reshape(-1, 1), results)
            base_est = lr.predict([[0.0]])[0]
            
        return base_est

    def predict(self, qc, instructions):
        # 1. ZNE Baseline
        base_estimate = self.run_zne(qc)
        
        predicted_residual = 0.0
        if self.model:
            # 2. AI Correction
            seq = self.tokenizer.tokenize(instructions)
            seq_t = torch.tensor([seq], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                predicted_residual = self.model(seq_t).item()
            
        final_prediction = base_estimate + predicted_residual
        return final_prediction, predicted_residual, base_estimate

    def get_ground_truth(self, qc):
        """Simulates ideal circuit."""
        sim_ideal = AerSimulator(method='stabilizer')
        try:
            res = sim_ideal.run(transpile(qc, sim_ideal), shots=1000).result().get_counts()
        except:
            sim_ideal = AerSimulator(method='statevector')
            res = sim_ideal.run(transpile(qc, sim_ideal), shots=1000).result().get_counts()
            
        shots = sum(res.values())
        true_val = (res.get('00',0)+res.get('11',0) - res.get('01',0)-res.get('10',0))/shots
        return true_val, res
