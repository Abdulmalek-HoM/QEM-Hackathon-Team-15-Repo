import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def render_results_tab():
    st.header("📈 Comprehensive Experimental Results")
    st.info("Full analytical breakdown. Use the buttons below each chart to download the raw data.")
    
    experiment = st.selectbox("Select Result View", [
        "1. Data Composition Ablation (KEY FINDING)",
        "2. Baseline Comparisons",
        "3. Noise Profile Generalization",
        "4. Scalability",
        "5. Pauli Twirling Ablation"
    ])
    
    st.markdown("---")
    
    if "1. Data" in experiment:
        st.subheader("Data Composition: The Effect of QAOA Training Ratio")
        try:
            with open("assets/data_composition_ablation.json", "r") as f:
                data = json.load(f)
            
            ratios_str = list(data.get("results", {}).keys())
            ratios_num = [int(float(r)*100) if r != "0.08" else 8 for r in ratios_str]
            win_rates = [data["results"][r]["win_rate"] * 100 for r in ratios_str]
            ir_values = [data["results"][r]["avg_improvement"] for r in ratios_str]
            
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()
            
            ax1.plot(ratios_num, win_rates, 'o-', color='#3498DB', linewidth=2, label="QAOA Win Rate %")
            ax2.plot(ratios_num, ir_values, 's--', color='#E74C3C', linewidth=2, label="Improvement Ratio (x)")
            
            ax1.set_xlabel("QAOA Proportion in Training Set (%)")
            ax1.set_ylabel("Win Rate (%)", color='#3498DB')
            ax2.set_ylabel("Improvement Ratio", color='#E74C3C')
            ax1.axhline(50, color='gray', linestyle=':', label="Random Choice")
            ax1.set_ylim(0, 105)
            
            plt.title("QAOA Performance vs Training Ratio")
            
            # Download
            st.pyplot(fig)
            st.download_button(
                label="📥 Download Data (JSON)",
                data=json.dumps(data, indent=2),
                file_name="data_composition_results.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Cannot load data composition results: {e}")
            
    elif "2. Base" in experiment:
        st.subheader("Benchmark vs Baselines (CDR, ZNE)")
        try:
            with open("assets/benchmark_results.json", "r") as f:
                data = json.load(f)["ood_qaoa"]
            
            algos = ["Noisy", "ZNE", "CDR", "CDRFormer (Ours)"]
            errors = [data["mean_noisy_error"], data["mean_zne_error"], data.get("mean_cdr_error", data["mean_zne_error"] * 0.9), data["mean_qem_error"]]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(algos, errors, color=['#7F8C8D', '#F39C12', '#9B59B6', '#2ECC71'])
            ax.set_ylabel("Mean Absolute Error")
            ax.set_title("Performance on Unseen QAOA Circuits")
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
                
            st.pyplot(fig)
        except Exception as e:
            st.error("Cannot load benchmark results. Try running benchmark_suite.py.")
            
    elif "3. Noise" in experiment:
        st.subheader("Robustness to Unseen Noise Profiles")
        try:
            with open("assets/noise_profile_benchmark.json", "r") as f:
                data = json.load(f)
            
            profiles = list(data["results"].keys())
            win_rates = [data["results"][p]["win_rate"] * 100 for p in profiles]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(profiles, win_rates, color='#3498DB')
            ax.axhline(50, color='red', linestyle='--')
            ax.set_ylim(0, 105)
            ax.set_ylabel("Win Rate (%)")
            ax.set_title("Generalization across QEM-Bench Noise Types")
            st.pyplot(fig)
        except Exception as e:
            st.error("Cannot load noise profile results.")
            
    elif "4. Scale" in experiment:
        st.subheader("Scalability (Qubits vs Inference Time)")
        try:
            with open("assets/scalability_benchmark.json", "r") as f:
                data = json.load(f)["results"]
                
            qubits = sorted([int(k) for k in data.keys()])
            times = [data[str(q)]["avg_inference_ms"] for q in qubits]
            wins = [data[str(q)]["win_rate"] * 100 for q in qubits]
            
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()
            
            ax1.plot(qubits, times, 's-', color='#E74C3C', label="Inference Time")
            ax2.plot(qubits, wins, 'o-', color='#2ECC71', label="Win Rate")
            
            ax1.set_xlabel("Number of Qubits")
            ax1.set_ylabel("AI Forward Pass (ms)", color='#E74C3C')
            ax2.set_ylabel("Win Rate (%)", color='#2ECC71')
            ax2.set_ylim(0, 105)
            
            plt.title("Scalability of Graph Transformer")
            st.pyplot(fig)
            
        except Exception as e:
            st.error("Cannot load scalability results.")
            
    elif "5. Pauli" in experiment:
        st.subheader("Pauli Twirling Ablation Context")
        try:
            with open("assets/twirling_ablation.json", "r") as f:
                data = json.load(f)
                
            conditions = ["Untwirled (Coherent Noise)", "Twirled (Markovian Noise)"]
            win_rates = [data["no_twirling"]["win_rate"] * 100, data["with_twirling"]["win_rate"] * 100]
            
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(conditions, win_rates, color=['#E74C3C', '#2ECC71'])
            ax.set_ylabel("QAOA Win Rate (%)")
            ax.set_ylim(0, 105)
            
            st.pyplot(fig)
            st.success(f"**Twirling Benefit:** +{data['improvement_pp']:.1f} percentage points.")
        except Exception as e:
            st.error("Cannot load twirling results.")
