from pydantic import BaseModel
from typing import List, Optional, Dict

class CircuitRequest(BaseModel):
    qasm_str: str
    model_type: str = "lstm" # or "transformer"

class MetricsResponse(BaseModel):
    true_val: float
    zne_base: float
    final_pred: float
    improvement_ratio: float
    ai_residual: float

class SimulationResponse(BaseModel):
    circuit_name: str
    metrics: MetricsResponse
    counts: Dict[str, int]
