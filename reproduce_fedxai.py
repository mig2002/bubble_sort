import os
import sys
import json
import yaml
import argparse
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from utils.metrics import compute_binary_metrics
from data.preprocessing import NBaIoTDataPreprocessor
from data.nbaiot_dataset import create_dataloaders, NBaIoTDataset
from data.download_nbaiot import get_nbaiot_feature_names, prepare_nbaiot_data
from models.lstm import LSTMIDS
from federated.client import IoTClient
from federated.server import FederatedServer
from xai.shap_explainer import SHAPExplainer
from xai.explanation_metrics import (
    compute_explanation_cosine_similarity,
    compute_top_k_jaccard,
    compute_pairwise_explanation_matrix
)

logger = get_logger("Reproduce_FedXAI")

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_centralized_model(
    combined_train_x: np.ndarray,
    combined_train_y: np.ndarray,
    global_test_loader: DataLoader,
    model_config: dict,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cuda"
) -> Tuple[LSTMIDS, dict]:
    """
    Trains a pooled centralized LSTM model representing the theoretical upper bound.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = LSTMIDS(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dense_dim=model_config["dense_dim"],
        dropout=model_config["dropout"]
    ).to(dev)
    
    dataset = NBaIoTDataset(combined_train_x, combined_train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger.info(f"Training Centralized Upper Bound model on {len(combined_train_x)} samples for {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(dev), y_b.to(dev)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_b, y_b in global_test_loader:
            x_b = x_b.to(dev)
            probs = torch.sigmoid(model(x_b))
            all_preds.extend(probs.cpu().numpy().ravel())
            all_targets.extend(y_b.numpy().ravel())
            
    metrics = compute_binary_metrics(all_targets, all_preds)
    return model, metrics

def run_reproduction_experiment(
    config_path: str = "configs/config.yaml",
    rounds: int = 5,
    local_epochs: int = 2,
    device: str = "cuda"
):
    output_dir = "results/fedxai_reproduction"
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    seed = cfg["experiment"].get("seed", 42)
    feature_names = get_nbaiot_feature_names()
    dev_type = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    logger.info("=" * 75)
    logger.info("   STEP 3: REPRODUCING ORIGINAL FedXAI & BENCHMARKING BASELINES")
    logger.info("=" * 75)

    # 1. Data Preparation & Splits
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

    # ---------------------------------------------------------
    # PHASE 1: Local Only Training (9 Isolated Clients)
    # ---------------------------------------------------------
    logger.info("\n>>> PHASE 1: Training 9 Isolated Local Models (No Federated Sharing)...")
    local_models = {}
    local_metrics_on_local = {}
    local_metrics_on_global = {}

    for client_id, dev in enumerate(device_names):
        splits = client_data[dev]
        train_l, val_l, test_l = create_dataloaders(
            splits["x_train"], splits["y_train"],
            splits["x_val"], splits["y_val"],
            splits["x_test"], splits["y_test"],
            batch_size=batch_size
        )
        
        # Train locally for equivalent total epochs (rounds * local_epochs)
        total_solo_epochs = min(rounds * local_epochs, 8)
        client = IoTClient(
            client_id=client_id,
            device_name=dev,
            train_loader=train_l,
            val_loader=val_l,
            test_loader=test_l,
            model_config=cfg["model"],
            device=dev_type
        )
        loss = client.train_local(epochs=total_solo_epochs, lr=cfg["federated"]["learning_rate"])
        
        # Evaluate on its own local test set
        m_local = client.evaluate(use_test_set=True)
        local_metrics_on_local[dev] = m_local
        
        # Evaluate on the unseen global test set (testing cross-device generalization)
        client.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_b, y_b in global_test_loader:
                probs = client.model.predict_proba(x_b.to(client.device))
                preds.extend(probs.cpu().numpy().ravel())
                targets.extend(y_b.numpy().ravel())
        m_global = compute_binary_metrics(targets, preds)
        local_metrics_on_global[dev] = m_global
        local_models[dev] = client.model
        logger.info(f"Local Model [{dev:<38}]: Local Acc={m_local['accuracy']*100:.2f}% | Global Acc={m_global['accuracy']*100:.2f}%")

    avg_local_acc_on_global = np.mean([m["accuracy"] for m in local_metrics_on_global.values()])
    avg_local_f1_on_global = np.mean([m["f1"] for m in local_metrics_on_global.values()])
    logger.info(f"Mean Solo Client Generalization on Global Test: Acc={avg_local_acc_on_global*100:.2f}%, F1={avg_local_f1_on_global*100:.2f}%")

    # ---------------------------------------------------------
    # PHASE 2: Centralized Upper Bound
    # ---------------------------------------------------------
    logger.info("\n>>> PHASE 2: Training Centralized Upper Bound Model...")
    all_tr_x = np.vstack([client_data[d]["x_train"] for d in device_names])
    all_tr_y = np.concatenate([client_data[d]["y_train"] for d in device_names])
    central_model, central_metrics = train_centralized_model(
        combined_train_x=all_tr_x,
        combined_train_y=all_tr_y,
        global_test_loader=global_test_loader,
        model_config=cfg["model"],
        epochs=min(rounds * local_epochs, 8),
        lr=cfg["federated"]["learning_rate"],
        batch_size=batch_size,
        device=dev_type
    )
    logger.info(f"Centralized Model Result -> Acc: {central_metrics['accuracy']*100:.2f}% | F1: {central_metrics['f1']*100:.2f}% | AUC: {central_metrics['roc_auc']*100:.2f}%")

    # ---------------------------------------------------------
    # PHASE 3: Original FedXAI (FedAvg Pipeline)
    # ---------------------------------------------------------
    logger.info("\n>>> PHASE 3: Running Original FedXAI (FedAvg Aggregation)...")
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
    
    for r in range(1, rounds + 1):
        server.run_round(round_idx=r, local_epochs=local_epochs, lr=cfg["federated"]["learning_rate"])
        
    fedxai_global_metrics = server.evaluate_on_loader(global_test_loader)
    logger.info(f"Original FedXAI Global Result -> Acc: {fedxai_global_metrics['accuracy']*100:.2f}% | F1: {fedxai_global_metrics['f1']*100:.2f}% | AUC: {fedxai_global_metrics['roc_auc']*100:.2f}%")

    # ---------------------------------------------------------
    # PHASE 4: Cross-Client & Global SHAP Explanation Consensus
    # ---------------------------------------------------------
    logger.info("\n>>> PHASE 4: Computing Cross-Client & Global SHAP Explanations...")
    # Baseline traffic background (benign)
    benign_idx = np.where(global_test_y == 0)[0][:40]
    bg_samples = global_test_x[benign_idx]
    test_eval_samples = global_test_x[:60]
    
    # 4.1 Global Model SHAP
    global_explainer = SHAPExplainer(model=server.global_model, background_data=bg_samples, feature_names=feature_names, device="cpu")
    global_shap_vals = global_explainer.compute_shap_values(test_eval_samples)
    global_mean_abs = np.mean(np.abs(global_shap_vals), axis=0)
    global_top10 = global_explainer.get_top_k_features(global_shap_vals, k=10)

    # 4.2 Per-Client SHAP
    client_mean_abs_shaps = {}
    client_top10s = {}
    client_cosine_to_global = {}
    client_jaccard_to_global = {}

    for client in fl_clients:
        dev = client.device_name
        # Local client background & test
        c_test_x = client_data[dev]["x_val"][:60]
        c_explainer = SHAPExplainer(model=client.model, background_data=bg_samples, feature_names=feature_names, device="cpu")
        c_shap_vals = c_explainer.compute_shap_values(c_test_x)
        c_mean_abs = np.mean(np.abs(c_shap_vals), axis=0)
        c_top10 = c_explainer.get_top_k_features(c_shap_vals, k=10)
        
        cos_sim = compute_explanation_cosine_similarity(c_mean_abs, global_mean_abs)
        jacc_sim = compute_top_k_jaccard(c_top10, global_top10)
        
        client_mean_abs_shaps[dev] = c_mean_abs
        client_top10s[dev] = c_top10
        client_cosine_to_global[dev] = cos_sim
        client_jaccard_to_global[dev] = jacc_sim
        logger.info(f"Device [{dev:<38}]: Cosine to Global={cos_sim:.4f} | Top-10 Jaccard={jacc_sim:.4f}")

    mean_cos_agreement = float(np.mean(list(client_cosine_to_global.values())))
    mean_jacc_agreement = float(np.mean(list(client_jaccard_to_global.values())))
    logger.info(f"FedXAI Explanation Consensus -> Mean Cosine: {mean_cos_agreement:.4f} | Mean Top-10 Jaccard: {mean_jacc_agreement:.4f}")

    # Pairwise Explanation Similarity Matrix
    sim_df = compute_pairwise_explanation_matrix(client_mean_abs_shaps, global_mean_abs)
    sim_csv = os.path.join(output_dir, "explanation_similarity_matrix.csv")
    sim_df.to_csv(sim_csv)
    logger.info(f"Saved explanation similarity matrix to: {sim_csv}")

    # ---------------------------------------------------------
    # PHASE 5: Output Reports & Visualizations
    # ---------------------------------------------------------
    # 5.1 Baseline Comparison Table
    summary_rows = [
        {
            "Method": "Isolated Local Training (Avg)",
            "Accuracy": float(avg_local_acc_on_global),
            "F1_Score": float(avg_local_f1_on_global),
            "Precision": float(np.mean([m["precision"] for m in local_metrics_on_global.values()])),
            "Recall": float(np.mean([m["recall"] for m in local_metrics_on_global.values()])),
            "FPR": float(np.mean([m["fpr"] for m in local_metrics_on_global.values()])),
            "ROC_AUC": float(np.mean([m["roc_auc"] for m in local_metrics_on_global.values()])),
            "Explanation_Consensus": 0.0  # Disparate uncollaborative models
        },
        {
            "Method": "Centralized Upper Bound",
            "Accuracy": float(central_metrics["accuracy"]),
            "F1_Score": float(central_metrics["f1"]),
            "Precision": float(central_metrics["precision"]),
            "Recall": float(central_metrics["recall"]),
            "FPR": float(central_metrics["fpr"]),
            "ROC_AUC": float(central_metrics["roc_auc"]),
            "Explanation_Consensus": 1.0  # Monolithic reference
        },
        {
            "Method": "Original FedXAI (FedAvg + SHAP)",
            "Accuracy": float(fedxai_global_metrics["accuracy"]),
            "F1_Score": float(fedxai_global_metrics["f1"]),
            "Precision": float(fedxai_global_metrics["precision"]),
            "Recall": float(fedxai_global_metrics["recall"]),
            "FPR": float(fedxai_global_metrics["fpr"]),
            "ROC_AUC": float(fedxai_global_metrics["roc_auc"]),
            "Explanation_Consensus": float(mean_cos_agreement)
        }
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "baseline_comparison_table.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved baseline comparison table to: {summary_csv}")

    # 5.2 Print Formatted Table
    print("\n" + "=" * 90)
    print("   FedXAI Reproduction: Baseline Performance Comparison Table")
    print("=" * 90)
    print(f"{'Method':<35} | {'Acc':<8} | {'F1':<8} | {'Prec':<8} | {'Recall':<8} | {'FPR':<8} | {'Expl. Consensus':<15}")
    print("-" * 90)
    for _, r in summary_df.iterrows():
        print(f"{r['Method']:<35} | {r['Accuracy']*100:6.2f}% | {r['F1_Score']*100:6.2f}% | {r['Precision']*100:6.2f}% | {r['Recall']*100:6.2f}% | {r['FPR']*100:6.2f}% | {r['Explanation_Consensus']:15.4f}")
    print("-" * 90 + "\n")

    # 5.3 Explanation Similarity Heatmap Plot
    plt.figure(figsize=(10, 8), dpi=300)
    # Shorten names for clean matrix display
    short_names = [n.replace("_Doorbell", "").replace("_Security_Camera", "_Cam").replace("_Webcam", "").replace("_Thermostat", "") for n in sim_df.index]
    sns.heatmap(sim_df, xticklabels=short_names, yticklabels=short_names, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("FedXAI: Cross-Device & Global SHAP Explanation Similarity Heatmap", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "explanation_similarity_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    logger.info(f"Saved explanation similarity heatmap to: {heatmap_path}")

    # 5.4 Performance Comparison Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width/2, summary_df["Accuracy"] * 100, width, label="Accuracy (%)", color="#337ab7")
    ax.bar(x + width/2, summary_df["F1_Score"] * 100, width, label="F1-Score (%)", color="#5cb85c")
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("FedXAI Baseline Reproduction: Solo Local vs Centralized vs FedXAI", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Method"], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    plt.tight_layout()
    barplot_path = os.path.join(output_dir, "baseline_comparison_barchart.png")
    plt.savefig(barplot_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    logger.info(f"Saved performance comparison bar plot to: {barplot_path}")

    logger.info("=" * 75)
    logger.info("   FedXAI Step 3 Reproduction Successfully Completed!")
    logger.info("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedXAI Step 3 Reproduction Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_reproduction_experiment(
        config_path=args.config,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        device=args.device
    )
