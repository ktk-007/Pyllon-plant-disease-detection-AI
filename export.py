import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import timm

from utils import MODEL_CONFIGS, PLANT_CONFIGS

def get_file_size_mb(path: Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)

def export_model(args):
    plant = args.plant
    model_type = args.model
    
    plant_cfg = PLANT_CONFIGS[plant]
    model_cfg = MODEL_CONFIGS[model_type]
    num_classes = plant_cfg["num_classes"]
    
    # Paths
    best_checkpoint_path = Path("models") / f"best_{model_type}_{plant}.pth"
    export_path = Path("models") / f"{model_type}_{plant}.pth"
    
    if not best_checkpoint_path.exists():
        print(f"Error: Could not find training checkpoint at {best_checkpoint_path}")
        print("Please train the model first.")
        return
        
    print(f"Exporting {model_type} for {plant}...")
    
    # 1. Load Base Architecture
    model = timm.create_model(
        model_cfg["timm_name"],
        pretrained=False,
        num_classes=num_classes
    )
    
    # 2. Load Weights (Trained unquantized weights)
    print(f"Loading weights from {best_checkpoint_path}")
    state_dict = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Apply Dynamic Quantization
    print("Applying dynamic quantization to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    # 4. Save Quantized Weights
    print(f"Saving quantized weights to {export_path}")
    torch.save(quantized_model.state_dict(), export_path)
    
    # 5. Print File Sizes
    orig_size = get_file_size_mb(best_checkpoint_path)
    quant_size = get_file_size_mb(export_path)
    
    print("-" * 40)
    print("EXPORT SUMMARY")
    print("-" * 40)
    print(f"Original size:  {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Reduction:      {(1 - quant_size/orig_size)*100:.1f}%")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", type=str, required=True, choices=list(PLANT_CONFIGS.keys()))
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()
    
    export_model(args)
