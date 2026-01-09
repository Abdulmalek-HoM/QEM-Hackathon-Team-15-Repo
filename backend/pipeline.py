import torch
import numpy as np
import sys
import os

# Add root directory to path to allow importing from utils and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.qem_former import QEMFormer
from data_gen_advanced import QEMGraphBuilder
import utils
from qiskit_aer import AerSimulator
from qiskit import transpile
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

class HackathonPipeline:
    def __init__(self, model_path="qem_former.pth"):
        self.device = torch.device('cpu') # Force CPU for safety in dashboard usually
        self.graph_builder = QEMGraphBuilder()
        
        # Load AI Model (QEM-Former)
        try:
            self.model = QEMFormer().to(self.device)
            # Load weights (weights_only=False for safety if trusted, or True if properly saved)
            # We saved with defaults in train_qem, which might require weights_only=False if global issues persist
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✅ AI Model (QEM-Former) Loaded Successfully.")
        except FileNotFoundError:
            print(f"⚠️ Model file {model_path} not found. AI correction will be disabled.")
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
            # Calculate expectation value <Z_0> (first qubit)
            # bitstring is qn..q0. q0 is last char.
            z0_val = 0
            for b, c in counts.items():
                val = 1 if b[-1] == '0' else -1
                z0_val += val * c
            val = z0_val / shots
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
        
        # 2. AI Prediction (Mitigated Value directly OR Residual?)
        # Our QEMFormer was trained to predict the *Ideal Value* directly from (NoisyGraph).
        # Wait, let's check train_qem.py / data_gen.
        # graph_data.y = ideal.
        # Input global_attr = [noisy_val, n_q, depth]. 
        # So model predicts Ideal. It's NOT a residual model in the strict sense (Base + Residual),
        # but the dashboard expects (final, residual, base).
        # We can define residual = predicted_ideal - base_estimate.
        
        final_prediction = base_estimate # Default if no model
        predicted_residual = 0.0
        
        if self.model:
            # Prepare Noisy Value for Context
            # We can re-use base_estimate (ZNE) or raw noisy. 
            # Models trained on raw noisy in context.
            # Let's run raw noisy quickly or extract from ZNE run? 
            # ZNE code runs scale 1.0. Let's optimize run_zne to return it? 
            # Or just re-run/use base_estimate (which is ZNE extrapolated). 
            # BUT: Model expects [RawNoisy, Qubits, Depth].
            # Training data used Sim_Noisy (Scale 1.5).
            # Here we should prob feed the equivalent.
            
            # Run noisy simulation to get context features
            nm = utils.build_noise_model(scale=1.0)  # Baseline noise
            sim = AerSimulator(noise_model=nm)
            
            t_qc = transpile(qc, sim)
            counts = sim.run(t_qc, shots=1000).result().get_counts()
            shots = sum(counts.values())
            
            # Calculate Z0 expectation
            z0_val = 0
            for b, c in counts.items():
                val = 1 if b[-1] == '0' else -1
                z0_val += val * c
            z0_noisy = z0_val / shots
            
            # Calculate ZZ correlation (Z0*Z1) if >= 2 qubits
            n_q = qc.num_qubits
            zz_noisy = 0.0
            if n_q >= 2:
                for b, c in counts.items():
                    try:
                        z0 = 1 if b[-1] == '0' else -1
                        z1 = 1 if b[-2] == '0' else -1
                        zz_noisy += z0 * z1 * c
                    except IndexError:
                        continue
                zz_noisy = zz_noisy / shots
            
            # Build Graph with 5-dim global context: [z0_noisy, zz_noisy, n_qubits, depth, noise_scale]
            depth = qc.depth()
            noise_scale = 1.0  # We use baseline noise for inference
            global_attr = [z0_noisy, zz_noisy, float(n_q), float(depth), noise_scale]
            
            graph = self.graph_builder.circuit_to_graph(qc, global_features=global_attr).to(self.device)
            
            with torch.no_grad():
                # Batch of 1
                batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(self.device)
                pred = self.model(graph.x, graph.edge_index, batch, graph.global_attr.unsqueeze(0))
                final_prediction = pred.item()
                
            predicted_residual = final_prediction - base_estimate
            
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
        # Calculate <Z_0>
        z0_val = 0
        for b, c in res.items():
            val = 1 if b[-1] == '0' else -1
            z0_val += val * c
        return z0_val / shots, res
