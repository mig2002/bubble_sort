import os
import sys
import yaml
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from data.preprocessing import NBaIoTDataPreprocessor
from data.download_nbaiot import get_nbaiot_feature_names, prepare_nbaiot_data
from data.nbaiot_dataset import create_dataloaders, NBaIoTDataset
from models.lstm import LSTMIDS
from federated.client import IoTClient
from federated.server import FederatedServer
from federated.aggregation import trust_weighted_aggregation
from security.model_trust import (
    ModelTrustEvaluator,
    calculate_client_update
)
from security.explanation_trust import (
    ExplanationTrustEvaluator,
    get_shap_representation,
    calculate_shap_similarity,
    get_top_k_features,
    calculate_top_k_jaccard,
    calculate_explanation_fidelity,
    calculate_explanation_trust
)
from xai.shap_explainer import SHAPExplainer

logger = get_logger("StageD_Expl_Trust")

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_stage_d_experiment(
    config_path: str = "configs/config.yaml",
    rounds: int = 2,
    device: str = "cuda"
):
    output_dir = "results/explanation_trust"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cfg = load_config(config_path)
    seed = cfg["experiment"].get("seed", 42)
    feature_names = get_nbaiot_feature_names()
    dev_type = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    logger.info("=" * 95)
    logger.info("   STAGE D: EXPLANATION TRUST EVALUATION (Similarity, Top-K Jaccard & Fidelity)")
    logger.info("   Aggregation uses Model Trust ONLY; Explanation Trust is evaluated observationally.")
    logger.info("=" * 95)

    # 1. Load Data
    data_dir = cfg["dataset"]["data_dir"]
    prepare_nbaiot_data(data_dir=data_dir, sample_size=cfg["dataset"].get("sample_size_per_client", 3000), seed=seed)
    
    preprocessor = NBaIoTDataPreprocessor(
        data_dir=data_dir,
        train_ratio=cfg["dataset"]["train_ratio"],
        val_ratio=cfg["dataset"]["val_ratio"],
        test_ratio=cfg["dataset"]["test_ratio"],
        seed=seed
    )
    device_names = cfg["clients"]["device_names"]
    client_data, (global_test_x, global_test_y) = preprocessor.load_and_preprocess_all(device_names)
    
    batch_size = cfg["federated"]["batch_size"]
    global_test_dataset = NBaIoTDataset(global_test_x, global_test_y)
    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Background baseline samples for SHAP
    benign_idx = np.where(global_test_y == 0)[0][:40]
    bg_samples = global_test_x[benign_idx]
    
    # Controlled reference sample batch for SHAP evaluation across all clients
    eval_x_sample = global_test_x[:50]

    fl_clients = []
    for client_id, dev in enumerate(device_names):
        splits = client_data[dev]
        train_l, val_l, test_l = create_dataloaders(
            splits["x_train"], splits["y_train"],
            splits["x_val"], splits["y_val"],
            splits["x_test"], splits["y_test"],
            batch_size=batch_size
        )
        c = IoTClient(
            client_id=client_id,
            device_name=dev,
            train_loader=train_l,
            val_loader=val_l,
            test_loader=test_l,
            model_config=cfg["model"],
            device=dev_type
        )
        fl_clients.append(c)

    server = FederatedServer(
        model_config=cfg["model"],
        clients=fl_clients,
        global_test_loader=global_test_loader,
        device=dev_type
    )

    model_evaluator = ModelTrustEvaluator(w_direction=0.50, w_norm=0.20, w_layer=0.30, reference_strategy="leave_one_out")
    expl_evaluator = ExplanationTrustEvaluator(w_similarity=0.40, w_jaccard=0.30, w_fidelity=0.30, top_k=5, reference_strategy="leave_one_out")

    # =========================================================================
    # PART 1: Multi-Round Clean Baseline (Model Trust + Explanation Trust)
    # =========================================================================
    logger.info("\n>>> PART 1: Multi-Round Clean Baseline Evaluation...")
    records_clean = []

    for r in range(1, rounds + 1):
        global_weights_before = server.get_global_weights()
        for client in fl_clients:
            client.receive_global_weights(global_weights_before)
            
        client_updates = []
        sample_counts = []
        client_deltas = {}
        client_models = {}
        client_eval_x = {}
        
        for i, client in enumerate(fl_clients):
            loss = client.train_local(epochs=1, lr=cfg["federated"]["learning_rate"])
            w_i, n_i = client.get_model_update()
            client_updates.append(w_i)
            sample_counts.append(n_i)
            
            delta_i = calculate_client_update(global_weights_before, w_i)
            client_deltas[client.device_name] = delta_i
            client_models[client.device_name] = client.model
            client_eval_x[client.device_name] = eval_x_sample

        # Step A: Model Trust Calculation
        m_trust_scores, _ = model_evaluator.evaluate_all(client_deltas)
        m_trust_list = [m_trust_scores[c.device_name]["model_trust"] for c in fl_clients]

        # Step B: SHAP Calculation on Controlled Reference Batch
        logger.info(f"Computing local SHAP explanations for round {r}...")
        client_explanations = {}
        for client in fl_clients:
            explainer = SHAPExplainer(model=client.model, background_data=bg_samples, feature_names=feature_names, device=dev_type)
            shap_vals = explainer.compute_shap_values(eval_x_sample)
            # 1D feature attribution representation (mean absolute SHAP)
            client_explanations[client.device_name] = get_shap_representation(shap_vals)

        # Step C: Explanation Trust Evaluation
        e_trust_scores, _ = expl_evaluator.evaluate_all(
            client_explanations=client_explanations,
            client_models=client_models,
            x_eval_dict=client_eval_x,
            feature_names=feature_names,
            device=dev_type
        )

        print(f"\n" + "=" * 115)
        print(f"   Round {r}: Model Trust vs Explanation Trust (Clean Baseline)")
        print("=" * 115)
        header = f"{'Client ID':<10} | {'Device Name':<34} | {'Model Trust':<12} | {'SHAP Sim':<12} | {'Top-K Jacc':<12} | {'Fidelity':<12} | {'Expl Trust':<12}"
        print(header)
        print("-" * len(header))
        
        for i, client in enumerate(fl_clients):
            dev = client.device_name
            m_val = m_trust_scores[dev]["model_trust"]
            e_res = e_trust_scores[dev]
            print(
                f"{i:<10} | {dev:<34} | "
                f"{m_val:12.4f} | "
                f"{e_res['shap_similarity']:12.4f} | "
                f"{e_res['top_k_stability']:12.4f} | "
                f"{e_res['fidelity']:12.4f} | "
                f"{e_res['explanation_trust']:12.4f}"
            )
            records_clean.append({
                "round": r,
                "client_id": i,
                "device_name": dev,
                "shap_similarity": e_res["shap_similarity"],
                "top_k_stability": e_res["top_k_stability"],
                "fidelity": e_res["fidelity"],
                "explanation_trust": e_res["explanation_trust"],
                "model_trust": m_val,
                "top_k_features": ",".join(e_res["top_k_features"])
            })

        # Aggregation strictly uses Model Trust (Stage C behavior preserved)
        agg_weights, tw, _ = trust_weighted_aggregation(client_updates, sample_counts, m_trust_list)
        server.global_model.set_weights(agg_weights)

    df_clean = pd.DataFrame(records_clean)
    csv_path_clean = "results/explanation_trust.csv"
    df_clean.to_csv(csv_path_clean, index=False)
    logger.info(f"\nSaved Explanation Trust evaluations to: {csv_path_clean}")

    # =========================================================================
    # PART 2: Sanity Check — Controlled Decoy Explanation Perturbation
    # =========================================================================
    logger.info("\n>>> PART 2: Sanity Check (Decoy Explanation Manipulation)...")
    logger.info("Client 8 submits HONEST model weights, but MANIPULATED (decoy) SHAP explanations.")
    
    # Take clean explanations from Round 2
    perturbed_explanations = dict(client_explanations)
    honest_client_8_expl = perturbed_explanations[device_names[8]].copy()
    
    # Forge decoy explanation for Client 8: swap top-K features with random bottom features
    top_5_names, top_5_idx = get_top_k_features(honest_client_8_expl, feature_names, k=5)
    decoy_expl = honest_client_8_expl.copy()
    # Zero out top decisive features and boost arbitrary low-ranked features
    decoy_expl[top_5_idx] = 0.0
    decoy_expl[100:105] = np.max(honest_client_8_expl) * 1.5
    perturbed_explanations[f"[Decoy XAI] {device_names[8]}"] = decoy_expl
    del perturbed_explanations[device_names[8]]
    
    perturbed_models = dict(client_models)
    perturbed_models[f"[Decoy XAI] {device_names[8]}"] = client_models[device_names[8]]
    del perturbed_models[device_names[8]]
    
    perturbed_eval_x = dict(client_eval_x)
    perturbed_eval_x[f"[Decoy XAI] {device_names[8]}"] = client_eval_x[device_names[8]]
    del perturbed_eval_x[device_names[8]]

    pert_trust_scores, _ = expl_evaluator.evaluate_all(
        client_explanations=perturbed_explanations,
        client_models=perturbed_models,
        x_eval_dict=perturbed_eval_x,
        feature_names=feature_names,
        device=dev_type
    )

    print("\n" + "=" * 120)
    print("   STAGE D SANITY CHECK: Honest IoT Devices vs Decoy Explanation Attacker")
    print("=" * 120)
    header_pert = f"{'Client Name':<45} | {'Category':<12} | {'Model Trust':<12} | {'SHAP Sim':<12} | {'Top-K Jacc':<12} | {'Fidelity':<12} | {'Expl Trust':<12}"
    print(header_pert)
    print("-" * len(header_pert))
    
    records_pert = []
    for name, res in pert_trust_scores.items():
        is_decoy = "[Decoy XAI]" in name
        cat = "Decoy XAI" if is_decoy else "Honest"
        # Client 8's model trust remains honest (~0.95)!
        m_score = m_trust_scores[device_names[8]]["model_trust"] if is_decoy else m_trust_scores[name]["model_trust"]
        print(
            f"{name:<45} | {cat:<12} | "
            f"{m_score:12.4f} | "
            f"{res['shap_similarity']:12.4f} | "
            f"{res['top_k_stability']:12.4f} | "
            f"{res['fidelity']:12.4f} | "
            f"{res['explanation_trust']:12.4f}"
        )
        records_pert.append({
            "Client": name,
            "Category": cat,
            "Model_Trust": m_score,
            "SHAP_Similarity": res["shap_similarity"],
            "Top_K_Stability": res["top_k_stability"],
            "Fidelity": res["fidelity"],
            "Explanation_Trust": res["explanation_trust"]
        })
    print("-" * len(header_pert) + "\n")

    df_pert = pd.DataFrame(records_pert)
    df_pert.to_csv(os.path.join(output_dir, "explanation_trust_decoy_check.csv"), index=False)

    # Render Stage D Visualization
    render_stage_d_plot(df_clean, df_pert, output_dir)
    return df_clean, df_pert

def render_stage_d_plot(df_clean: pd.DataFrame, df_pert: pd.DataFrame, output_dir: str):
    """Renders Stage D diagnostic comparison charts."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Model Trust vs Explanation Trust across Honest Clients
    df_r2 = df_clean[df_clean["round"] == df_clean["round"].max()]
    x = np.arange(len(df_r2))
    width = 0.35
    
    axes[0].bar(x - width/2, df_r2["model_trust"], width, label="Model Trust $M_i$", color="#3182bd", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + width/2, df_r2["explanation_trust"], width, label="Explanation Trust $E_i$", color="#756bb1", edgecolor="black", linewidth=0.5)
    
    axes[0].set_title("A. Clean Baseline: Model Trust vs Explanation Trust", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"C{i}" for i in range(len(df_r2))], rotation=0)
    axes[0].set_ylabel("Trust Score $\in [0, 1]$", fontweight="bold")
    axes[0].set_ylim(0.0, 1.15)
    axes[0].legend(loc="lower right")

    # Panel 2: Sanity Check — Case 3 Discovery (High Model Trust + Low Explanation Trust)
    colors = ["#756bb1" if cat == "Honest" else "#de2d26" for cat in df_pert["Category"]]
    clean_names = [n.replace("[Decoy XAI] ", "") for n in df_pert["Client"]]
    bars = axes[1].barh(clean_names[::-1], df_pert["Explanation_Trust"][::-1], color=colors[::-1], edgecolor="black", linewidth=0.6)
    
    for bar in bars:
        w = bar.get_width()
        axes[1].text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.3f}", ha="left", va="center", fontsize=9, fontweight="bold")
        
    axes[1].set_title("B. Sanity Check: Decoy XAI Detection (Red = Decoy Explainer)", fontweight="bold")
    axes[1].set_xlabel("Explanation Trust Score $E_i$", fontweight="bold")
    axes[1].set_xlim(0.0, 1.15)
    axes[1].axvline(0.5, color="gray", linestyle="--", label="Trust Decision Threshold (0.50)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "stage_d_explanation_trust_validation.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Stage D validation figure to: {chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage D: Explanation Trust Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_stage_d_experiment(config_path=args.config, rounds=args.rounds, device=args.device)
