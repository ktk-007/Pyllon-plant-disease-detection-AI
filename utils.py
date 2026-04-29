"""
utils.py — PlantAI Plant Disease Detector
Shared utilities: datasets, transforms, data loaders,
class-weight computation, MixUp, FocalLoss, and label saving.
"""

import os
import json
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
CLASS_LABELS_DIR = Path("class_labels")
CLASS_LABELS_DIR.mkdir(exist_ok=True)

# ImageNet normalisation statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Plant / model configuration tables
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict] = {
    "convnext": {"timm_name": "convnext_tiny", "lr": 2e-4, "weight_decay": 0.05, "head_attr": "head"},
    "effnet":   {"timm_name": "tf_efficientnetv2_s", "lr": 2e-4, "weight_decay": 0.05, "head_attr": "classifier"},
}

PLANT_CONFIGS: Dict[str, Dict] = {
    "tomato":      {"num_classes": 6, "loss": "crossentropy"},
    "mango_leaf":  {"num_classes": 6, "loss": "crossentropy"},
    "mango_fruit": {"num_classes": 5, "loss": "crossentropy"},
    "apple":       {"num_classes": 4, "loss": "crossentropy"},
    "potato":      {"num_classes": 3, "loss": "crossentropy"},
    "rose":        {"num_classes": 5, "loss": "focal"},
    "corn":        {"num_classes": 4, "loss": "crossentropy"},
    "bellpepper":  {"num_classes": 2, "loss": "crossentropy"},
    "grape":       {"num_classes": 4, "loss": "crossentropy"},
    "strawberry":  {"num_classes": 2, "loss": "crossentropy"},
}

# ---------------------------------------------------------------------------
# Mango leaf / fruit split logic
# ---------------------------------------------------------------------------
# Leaf model: classes that have LF or BG images
# Fruit model: classes that have only FR images
# mango__anthracnose and mango__healthy appear in BOTH datasets,
# but we keep them together in each split by filtering on file prefix.

# Folder names in ImageFolder that belong to each mango model.
# Key = canonical plant name used in training scripts.
MANGO_LEAF_CLASSES = [
    "mango__anthracnose",        # LF + BG files
    "mango__bacterial_canker",   # LF + BG files
    "mango__gall_midge",         # LF + BG files
    "mango__healthy",            # LF + BG files (leaf split)
    "mango__powdery_mildew",     # LF + BG files
    "mango__sooty_mould",        # LF + BG files
]

MANGO_FRUIT_CLASSES = [
    "mango__alternaria",         # FR only
    "mango__anthracnose",        # FR files (fruit split)
    "mango__black_mould_rot",    # FR only
    "mango__healthy",            # FR files (fruit split)
    "mango__stem_end_rot",       # FR only
]

# File prefix that determines leaf vs fruit within a shared class
LEAF_PREFIXES  = ("MAN_LF_", "MAN_BG_")   # LF = real leaf, BG = bg-removed leaf
FRUIT_PREFIX   = "MAN_FR_"

# Which folder names to include per plant key (non-mango plants use prefix match)
PLANT_CLASS_PREFIXES: Dict[str, str] = {
    "tomato":      "tomato__",
    "mango_leaf":  "mango__",   # filtered further by file prefix
    "mango_fruit": "mango__",   # filtered further by file prefix
    "apple":       "apple__",
    "potato":      "potato__",
    "rose":        "rose__",
    "corn":        "corn__",
    "bellpepper":  "bell_pepper__",
    "grape":       "grape__",
    "strawberry":  "strawberry__",
}

# ---------------------------------------------------------------------------
# Albumentations Dataset wrapper
# ---------------------------------------------------------------------------

class AlbumentationsDataset(Dataset):
    """
    Wraps a torchvision ImageFolder with Albumentations transforms.
    Optionally filters to a subset of class indices and/or individual
    file samples (for the mango leaf/fruit split).
    """

    def __init__(
        self,
        root: str,
        transform: A.Compose,
        class_filter: Optional[List[str]] = None,
        file_filter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Args:
            root:         Path to an ImageFolder-style directory.
            transform:    Albumentations Compose pipeline.
            class_filter: If given, only include subdirs whose names are in
                          this list. Class indices are re-mapped to 0..N-1
                          in alphabetical order of the filtered names.
            file_filter:  Optional callable(filepath) → bool.  If given,
                          only samples whose file path satisfies the predicate
                          are included.  Used for mango LF/BG vs FR split.
        """
        # Build the full ImageFolder first to get all class names / samples
        full_dataset = ImageFolder(root=root)

        if class_filter is not None:
            # Keep only classes in class_filter, remap indices
            filtered_classes = sorted(
                [c for c in full_dataset.classes if c in class_filter]
            )
        else:
            filtered_classes = sorted(full_dataset.classes)

        # Build old-class-name → new-index mapping
        new_class_to_idx = {c: i for i, c in enumerate(filtered_classes)}
        old_class_to_idx = full_dataset.class_to_idx

        # Filter samples
        valid_old_indices = {
            old_class_to_idx[c] for c in filtered_classes if c in old_class_to_idx
        }

        filtered_samples = []
        for filepath, old_label in full_dataset.samples:
            if old_label not in valid_old_indices:
                continue
            if file_filter is not None and not file_filter(filepath):
                continue
            class_name = full_dataset.classes[old_label]
            new_label  = new_class_to_idx[class_name]
            filtered_samples.append((filepath, new_label))

        self.samples       = filtered_samples
        self.classes       = filtered_classes
        self.class_to_idx  = new_class_to_idx
        self.transform     = transform

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        img = cv2.imread(filepath)
        if img is None:
            # Fallback: use PIL for images cv2 can't open
            img = np.array(Image.open(filepath).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        return augmented["image"], label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str, plant: str = "") -> A.Compose:
    """
    Returns an Albumentations Compose pipeline.

    Args:
        split: "train" | "val" | "test"
        plant: plant name — used to add mango-specific augmentation.
    """
    is_mango = plant.startswith("mango")

    if split == "train":
        # For mango: Only pad if necessary to keep the leaf centered without excessive black space
        pre_crop = []
        if is_mango:
            pre_crop = [A.Pad(padding=10, border_mode=0, fill=0, p=1.0)]
        
        aug_list = pre_crop + [
            A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4,
            ),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
        return A.Compose(aug_list)

    else:  # val / test
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


# ---------------------------------------------------------------------------
# Mango file-level filter helpers
# ---------------------------------------------------------------------------

def _is_leaf_file(filepath: str) -> bool:
    """True if the image belongs to the *leaf* mango model (LF or BG prefix)."""
    fname = os.path.basename(filepath)
    return fname.startswith(LEAF_PREFIXES)


def _is_fruit_file(filepath: str) -> bool:
    """True if the image belongs to the *fruit* mango model (FR prefix)."""
    fname = os.path.basename(filepath)
    return fname.startswith(FRUIT_PREFIX)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def get_dataloaders(
    plant: str,
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for a given plant.

    Mango is handled as a special case: mango_leaf and mango_fruit
    are separated by file prefix so both models share the same folder
    structure but see different subsets of images.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_dir)

    # Determine class filter list and optional file-level filter
    if plant == "mango_leaf":
        class_filter = MANGO_LEAF_CLASSES
        file_filter  = _is_leaf_file
    elif plant == "mango_fruit":
        class_filter = MANGO_FRUIT_CLASSES
        file_filter  = _is_fruit_file
    else:
        prefix = PLANT_CLASS_PREFIXES[plant]
        # class_filter is built dynamically from what exists on disk
        class_filter = None   # will be resolved per split below
        file_filter  = None

    loaders = []
    for split in ("train", "val", "test"):
        split_root = str(data_path / split)

        # Resolve class filter for non-mango plants
        if plant not in ("mango_leaf", "mango_fruit"):
            prefix = PLANT_CLASS_PREFIXES[plant]
            existing = [
                d for d in os.listdir(split_root)
                if os.path.isdir(os.path.join(split_root, d)) and d.startswith(prefix)
            ]
            cf = sorted(existing)
        else:
            cf = class_filter

        transform = get_transforms(split, plant)
        dataset   = AlbumentationsDataset(
            root=split_root,
            transform=transform,
            class_filter=cf,
            file_filter=file_filter,
        )

        shuffle = (split == "train")
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),   # avoids 1-sample batches with MixUp
            persistent_workers=(num_workers > 0),
        )
        loaders.append(loader)

    return tuple(loaders)   # (train, val, test)


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: AlbumentationsDataset) -> torch.Tensor:
    """
    Compute balanced class weights from a dataset using sklearn.

    Returns a FloatTensor of shape (num_classes,).
    """
    labels = [label for _, label in dataset.samples]
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(labels),
    )
    return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------

def mixup_data(x, y, num_classes, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a_onehot = F.one_hot(y, num_classes=num_classes).float()
    y_b_onehot = F.one_hot(y[index], num_classes=num_classes).float()

    return mixed_x, y_a_onehot, y_b_onehot, lam


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a_onehot: torch.Tensor,
    y_b_onehot: torch.Tensor,
    lam: float,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute mixed-label loss compatible with CrossEntropy or FocalLoss.

    For CrossEntropy we apply label smoothing manually since we're working
    with soft targets.

    Args:
        criterion:       Loss object (FocalLoss or nn.CrossEntropyLoss).
                         For MixUp we bypass the criterion and compute the
                         soft-target cross-entropy directly.
        pred:            Model logits (B, C).
        y_a_onehot:      One-hot labels for sample A (B, C).
        y_b_onehot:      One-hot labels for sample B (B, C).
        lam:             Mixing coefficient.
        class_weights:   Per-class weights tensor or None.
        label_smoothing: Smoothing factor ε applied to soft targets.
        num_classes:     Number of classes (needed for smoothing).
    Returns:
        Scalar loss tensor.
    """
    num_cls = pred.size(1) if num_classes is None else num_classes

    # Soft mixed target
    mixed_targets = lam * y_a_onehot + (1.0 - lam) * y_b_onehot  # (B, C)

    # Apply label smoothing: (1-ε)*target + ε/C
    if label_smoothing > 0.0:
        mixed_targets = (
            (1.0 - label_smoothing) * mixed_targets
            + label_smoothing / num_cls
        )

    log_probs = F.log_softmax(pred, dim=1)  # (B, C)

    if class_weights is not None:
        # Weight each class contribution
        weights = class_weights.to(pred.device)                  # (C,)
        # Per-sample loss = -sum_c w_c * t_c * log_p_c
        loss = -(mixed_targets * log_probs * weights.unsqueeze(0)).sum(dim=1)
    else:
        loss = -(mixed_targets * log_probs).sum(dim=1)

    return loss.mean()


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(pt) = -(1 - pt)^gamma * log(pt)

    Args:
        alpha:  Per-class weight tensor (same role as class_weights in CE).
                Shape (num_classes,) or None.
        gamma:  Focusing parameter.
        reduction: "mean" | "sum" | "none".
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs:  Logits (B, C).
            targets: Integer class labels (B,)  OR soft one-hot (B, C).
        """
        if targets.dim() == 2:
            # Soft targets (MixUp path) — use KL-style focal approximation
            log_p  = F.log_softmax(inputs, dim=1)
            p      = torch.exp(log_p)
            # Effective pt = sum_c targets_c * p_c
            pt     = (targets * p).sum(dim=1)
            ce     = -(targets * log_p).sum(dim=1)
            if self.alpha is not None:
                # Weighted by expected class weight
                alpha_t = (targets * self.alpha.to(inputs.device)).sum(dim=1)
                ce = alpha_t * ce
            focal  = ((1.0 - pt) ** self.gamma) * ce
        else:
            # Hard targets
            alpha_tensor = self.alpha.to(inputs.device) if self.alpha is not None else None
            ce_loss  = F.cross_entropy(inputs, targets, weight=alpha_tensor, reduction="none")
            pt       = torch.exp(-ce_loss)
            focal    = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# Label saving
# ---------------------------------------------------------------------------

def save_class_labels(dataset: AlbumentationsDataset, plant: str) -> Path:
    """
    Save class labels to class_labels/{plant}_classes.json.

    The JSON stores a list of class names ordered by their integer index.

    Args:
        dataset: An AlbumentationsDataset (after filtering).
        plant:   Plant key string (e.g. "tomato", "mango_leaf").

    Returns:
        Path to the written JSON file.
    """
    CLASS_LABELS_DIR.mkdir(exist_ok=True)
    out_path = CLASS_LABELS_DIR / f"{plant}_classes.json"

    # idx → class_name list (sorted by index value)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    ordered = [idx_to_class[i] for i in range(len(idx_to_class))]

    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)

    print(f"[utils] Saved {len(ordered)} class labels -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Quick dataset info helper (used in train.py for sanity checks)
# ---------------------------------------------------------------------------

def dataset_info(dataset: AlbumentationsDataset, split: str, plant: str) -> None:
    """Print a summary of dataset composition."""
    from collections import Counter
    labels = [lbl for _, lbl in dataset.samples]
    counts = Counter(labels)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    total = len(dataset)
    print(f"\n[{split.upper()}] plant={plant}  total={total}")
    for idx in sorted(counts):
        name = idx_to_class[idx]
        print(f"  [{idx}] {name}: {counts[idx]}")
    print()


# ---------------------------------------------------------------------------
# Self-test (run `python utils.py` to verify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    plant = sys.argv[1] if len(sys.argv) > 1 else "tomato"
    print(f"=== utils.py self-test for plant: {plant} ===\n")

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            plant=plant,
            data_dir="data",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )
    except Exception as e:
        print(f"[ERROR] get_dataloaders failed: {e}")
        sys.exit(1)

    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    test_ds  = test_loader.dataset

    dataset_info(train_ds, "train", plant)
    dataset_info(val_ds,   "val",   plant)
    dataset_info(test_ds,  "test",  plant)

    # Save class labels
    save_class_labels(train_ds, plant)

    # Check class weights
    weights = compute_class_weights(train_ds)
    print(f"Class weights: {weights.tolist()}\n")

    # Check MixUp
    images, labels = next(iter(train_loader))
    nc = PLANT_CONFIGS[plant]["num_classes"]
    mixed_x, ya, yb, lam = mixup_data(images, labels, num_classes=nc)
    print(f"MixUp: lam={lam:.4f}  mixed_x shape={mixed_x.shape}")

    # Check FocalLoss
    logits = torch.randn(4, nc)
    fl = FocalLoss(alpha=weights, gamma=2.0)
    loss_val = fl(logits, labels)
    print(f"FocalLoss (hard targets): {loss_val.item():.4f}")

    loss_soft = fl(logits, ya)
    print(f"FocalLoss (soft targets): {loss_soft.item():.4f}")

    print("\n[utils.py] All checks passed ✓")
