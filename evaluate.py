import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import timm
from utils import get_dataloaders, MODEL_CONFIGS, PLANT_CONFIGS

# ---------------------------------------------------------------------------
# Setup and Model Loading
# ---------------------------------------------------------------------------

def load_eval_model(plant: str, model_type: str, device: torch.device):
    """
    Loads the trained best checkpoint for the given plant and model type,
    and applies dynamic quantization to match the deployment environment.
    """
    model_cfg = MODEL_CONFIGS[model_type]
    plant_cfg = PLANT_CONFIGS[plant]
    num_classes = plant_cfg["num_classes"]
    
    # Create base model
    model = timm.create_model(
        model_cfg["timm_name"],
        pretrained=False,
        num_classes=num_classes
    )
    
    # Load best checkpoint weights
    checkpoint_path = Path("models") / f"best_{model_type}_{plant}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Apply dynamic quantization to match what will run in Streamlit
    model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    model.to(device)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Evaluation Routine
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(model, loader, device):
    """Returns (all_preds, all_probs, all_targets)"""
    all_preds = []
    all_probs = []
    all_targets = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    return np.array(all_preds), np.array(all_probs), np.array(all_targets)

def top_k_accuracy(probs, targets, k=3):
    """Calculate Top-K accuracy."""
    if probs.shape[1] < k:
        k = probs.shape[1] # fallback if classes < k
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum([targets[i] in top_k_preds[i] for i in range(len(targets))])
    return correct / len(targets)

def plot_confusion_matrix(cm, class_names, save_path, title):
    """Plots and saves a confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def evaluate(args):
    plant = args.plant
    device = torch.device("cpu") # Using CPU to simulate Streamlit limits and quantization
    print(f"--- Evaluating {plant} on CPU (Simulating Production) ---")
    
    # 1. Load Data
    _, _, test_loader = get_dataloaders(plant, batch_size=32, num_workers=0)
    
    class_labels_file = Path("class_labels") / f"{plant}_classes.json"
    with open(class_labels_file, "r") as f:
        class_names = json.load(f)
        
    # Clean class names for display
    display_names = [name.split("__")[-1] for name in class_names]
        
    run_dir = Path("runs") / plant
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Models
    print("Loading models...")
    try:
        convnext = load_eval_model(plant, "convnext", device)
        effnet = load_eval_model(plant, "effnet", device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train BOTH models first before running evaluation.")
        return

    # 3. Get Predictions
    print("Running inference on ConvNeXt...")
    c_preds, c_probs, targets = get_predictions(convnext, test_loader, device)
    
    print("Running inference on EfficientNet...")
    e_preds, e_probs, _ = get_predictions(effnet, test_loader, device)
    
    # 4. Ensemble Predictions
    print("Calculating Ensemble...")
    # Weighted ensemble: 0.55 convnext + 0.45 effnet
    ens_probs = 0.55 * c_probs + 0.45 * e_probs
    ens_preds = np.argmax(ens_probs, axis=1)
    
    # 5. Metrics
    models_results = {
        "ConvNeXt": (c_preds, c_probs),
        "EfficientNet": (e_preds, e_probs),
        "Ensemble": (ens_preds, ens_probs)
    }
    
    report_text = f"=== Evaluation Report: {plant} ===\n\n"
    
    for name, (preds, probs) in models_results.items():
        top1 = accuracy_score(targets, preds) * 100
        top3 = top_k_accuracy(probs, targets, k=3) * 100
        
        print(f"\n[{name}]")
        print(f"Top-1 Accuracy: {top1:.2f}%")
        print(f"Top-3 Accuracy: {top3:.2f}%")
        
        report_text += f"--- {name} ---\n"
        report_text += f"Top-1 Accuracy: {top1:.2f}%\n"
        report_text += f"Top-3 Accuracy: {top3:.2f}%\n"
        
        if name == "Ensemble":
            # Full report only for ensemble
            clf_rep = classification_report(targets, preds, target_names=display_names, digits=4)
            print("\nClassification Report (Ensemble):")
            print(clf_rep)
            report_text += "\nClassification Report:\n"
            report_text += clf_rep + "\n"
            
            # Confusion Matrix
            cm = confusion_matrix(targets, preds)
            cm_path = run_dir / "ensemble_confusion_matrix.png"
            plot_confusion_matrix(cm, display_names, cm_path, f"{plant} Ensemble Confusion Matrix")
            print(f"Saved confusion matrix plot to {cm_path}")
            
    # Save text report
    report_path = run_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved full classification report to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", type=str, required=True, choices=list(PLANT_CONFIGS.keys()))
    args = parser.parse_args()
    
    evaluate(args)
