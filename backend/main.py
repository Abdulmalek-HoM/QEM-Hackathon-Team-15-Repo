import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qiskit import QuantumCircuit

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.schemas import CircuitRequest, SimulationResponse, MetricsResponse
from backend.pipeline import HackathonPipeline
import utils

app = FastAPI(title="QEM Hackathon API")

# Allow CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("🚀 Initializing QEM Pipeline...")
    # Assume model is at root
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'qem_lstm.pth'))
    pipeline = HackathonPipeline(lstm_path=model_path)

@app.get("/")
def read_root():
    return {"status": "active", "message": "QEM AI Backend is Running"}

@app.post("/simulate", response_model=SimulationResponse)
async def simulate_circuit(req: CircuitRequest):
    try:
        # Parse OpenQASM
        qc = QuantumCircuit.from_qasm_str(req.qasm_str)
        
        # Get instructions for AI
        # We need a robust way to get instructions list same as utils. 
        # For now, we rely on utils to help parse or we extract manually if needed.
        # But utils.create_random_clifford_circuit returns (qc, instructions).
        # Here we only have QC. 
        # REFACTOR: We need to reconstruct 'instructions' list from QC for the Tokenizer.
        # Let's add a helper in utils or here.
        
        # Simple extraction for now (Gateway to Tokenizer)
        instructions = []
        for instr in qc.data:
            # instr is (CircuitInstruction) = operation, qubits, clbits
            name = instr.operation.name
            qubits = [q._index for q in instr.qubits] 
            # Note: q._index is internal, cleaner is qc.find_bit(q).index
            # But for simple 2-qubit circuits in this hackathon, let's assume standard indices
            # Or use utils to extract.
            
            # Form: ['cx', 0, 1] or ['h', 0]
            item = [name] + qubits
            instructions.append(item)

        # Run Pipeline
        final_pred, ai_res, zne_base = pipeline.predict(qc, instructions)
        
        # Get Ground Truth for Metrics
        true_val, ideal_counts = pipeline.get_ground_truth(qc)
        
        # Calc Ratio
        err_base = abs(true_val - zne_base)
        err_model = abs(true_val - final_pred)
        
        ratio = 100.0 if err_model < 1e-5 else err_base / err_model
        
        metrics = MetricsResponse(
            true_val=true_val,
            zne_base=zne_base,
            final_pred=final_pred,
            improvement_ratio=ratio,
            ai_residual=ai_res
        )
        
        return SimulationResponse(
            circuit_name="Custom Circuit",
            metrics=metrics,
            counts=ideal_counts
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
