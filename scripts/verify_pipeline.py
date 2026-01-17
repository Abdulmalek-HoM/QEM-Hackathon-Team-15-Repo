
import os
import torch
from data_gen_advanced import generate_advanced_dataset, DATASET_DIR
from train_qem import train
from benchmark_suite import benchmark_models

def verify_pipeline():
    print("=== QEM Pipeline Verification ===")
    
    # 1. Clean previous data/models for fresh test
    print("\n[Step 1] Cleaning old artifacts...")
    if os.path.exists("weights/qem_former.pth"): os.remove("weights/qem_former.pth")
    # Don't delete dataset dir, just overwrite chunk 0
    
    # 2. Generate Data (Small Chunk)
    print("\n[Step 2] Generating Test Dataset...")
    generate_advanced_dataset(n_samples=50, min_qubits=3, max_qubits=5, chunk_id=999) # chunk 999 for test
    
    # Check if file exists
    if not os.path.exists(os.path.join(DATASET_DIR, "train_data_chunk_999.pt")):
        print("❌ Data Generation Failed!")
        return
    print("✅ Data Generation Success!")
    
    # 3. Train Model
    print("\n[Step 3] Training Logic-Aware Graph Transformer...")
    try:
        train()
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    if not os.path.exists("weights/qem_former.pth"):
        print("❌ Model file not found after training!")
        return
    print("✅ Training Success!")
    
    # 4. Benchmark
    print("\n[Step 4] Running Benchmark Suite...")
    try:
        benchmark_models("weights/qem_former.pth")
    except Exception as e:
        print(f"❌ Benchmarking Failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\n✅✅✅ VERIFICATION COMPLETE! The pipeline is operational. ✅✅✅")

if __name__ == "__main__":
    verify_pipeline()
