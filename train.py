"""
train.py — PlantAI Optimized Training Script
- AMP for RTX GPUs
- Saves checkpoint after every epoch (full state: model, optimizer, scheduler, EMA, scaler)
- Auto-resumes from last checkpoint if interrupted
"""

import os
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

import timm
from timm.utils import ModelEmaV2, AverageMeter
from timm.scheduler import CosineLRScheduler

from utils import (
    get_dataloaders,
    compute_class_weights,
    mixup_data,
    mixup_criterion,
    FocalLoss,
    MODEL_CONFIGS,
    PLANT_CONFIGS,
    save_class_labels,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CKPT_NAME = "last_checkpoint.pt"

def save_checkpoint(state: dict, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / (CKPT_NAME + ".tmp")
    torch.save(state, tmp)
    tmp.replace(ckpt_dir / CKPT_NAME)   # atomic rename — avoids corrupt files


def load_checkpoint(ckpt_dir: Path, model, ema_model, optimizer, scheduler, scaler, logger):
    ckpt_path = ckpt_dir / CKPT_NAME
    if not ckpt_path.exists():
        return 0, 0.0, 0   # start_epoch, best_acc, patience

    logger.info(f"Resuming from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    ema_model.module.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"] + 1
    best_acc    = ckpt["best_acc"]
    patience    = ckpt["patience"]
    logger.info(f"Resumed from epoch {ckpt['epoch']} | best_acc={best_acc:.2f}% | patience={patience}")
    return start_epoch, best_acc, patience

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    epoch, model, ema_model, loader, optimizer, criterion, device,
    num_classes, plant_cfg, class_weights, phase, logger,
    scaler=None, use_amp=False
):
    model.train()
    losses = AverageMeter()
    n_steps = len(loader)

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        if use_amp:
            with autocast(device_type="cuda"):
                if phase == 2:
                    mixed_x, ya, yb, lam = mixup_data(images, labels, num_classes)
                    out  = model(mixed_x)
                    loss = mixup_criterion(criterion, out, ya, yb, lam,
                                          class_weights=class_weights, num_classes=num_classes)
                else:
                    out  = model(images)
                    loss = criterion(out, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if phase == 2:
                mixed_x, ya, yb, lam = mixup_data(images, labels, num_classes)
                out  = model(mixed_x)
                loss = mixup_criterion(criterion, out, ya, yb, lam,
                                       class_weights=class_weights, num_classes=num_classes)
            else:
                out  = model(images)
                loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ema_model.update(model)
        losses.update(loss.item(), images.size(0))

        if i % 20 == 0 or i == n_steps - 1:
            logger.info(f"Epoch [{epoch}][{i}/{n_steps}] Loss {losses.val:.4f} ({losses.avg:.4f})")

    return losses.avg


@torch.no_grad()
def validate(model, loader, criterion, device, logger, plant_cfg):
    model.eval()
    top1 = AverageMeter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out  = model(images)
        acc1 = (out.argmax(1) == labels).float().mean() * 100
        top1.update(acc1.item(), images.size(0))
    logger.info(f"** Val EMA Acc@1: {top1.avg:.2f}%")
    return top1.avg

# ---------------------------------------------------------------------------
# Determine phase from epoch number
# ---------------------------------------------------------------------------

def get_phase(epoch):
    if epoch < 3:   return 1   # head warmup
    if epoch < 16:  return 2   # full finetune + MixUp
    return 3                    # polish


def set_frozen(model, freeze: bool):
    # If freeze is True, we only train the very last layer (head)
    # If freeze is False, we train everything.
    for param in model.parameters():
        param.requires_grad = not freeze
    
    # Always ensure the last layer is trainable
    if hasattr(model, 'head'):
        for param in model.head.parameters(): param.requires_grad = True
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters(): param.requires_grad = True
    if hasattr(model, 'fc'):
        for param in model.fc.parameters(): param.requires_grad = True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    plant_cfg = PLANT_CONFIGS[args.plant]
    model_cfg = MODEL_CONFIGS[args.model]
    num_classes = plant_cfg["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    pin = (device.type == "cuda")

    ckpt_dir  = Path("runs") / args.plant / args.model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    logger = setup_logger(ckpt_dir)
    writer = SummaryWriter(log_dir=str(ckpt_dir))
    logger.info(f"Training {args.plant}/{args.model} | device={device} | AMP={use_amp}")

    # ---- Data ----
    train_loader, val_loader, _ = get_dataloaders(
        args.plant, batch_size=args.batch_size, pin_memory=pin
    )
    save_class_labels(train_loader.dataset, args.plant)
    class_weights = compute_class_weights(train_loader.dataset).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # ---- Model ----
    model = timm.create_model(
        model_cfg["timm_name"], pretrained=True, num_classes=num_classes
    ).to(device)
    ema_model = ModelEmaV2(model, decay=0.9998)

    # ---- Loss ----
    if plant_cfg["loss"] == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ---- Optimizer & Scheduler ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"]
    )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        warmup_t=3,
        warmup_lr_init=1e-6,
        lr_min=1e-6
    )
    scaler = GradScaler("cuda") if use_amp else None

    # ---- Resume from checkpoint if exists ----
    start_epoch, best_acc, patience = load_checkpoint(
        ckpt_dir, model, ema_model, optimizer, scheduler, scaler, logger
    )

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        phase = get_phase(epoch)

        # Freeze / unfreeze backbone at boundary epochs
        if epoch == 0:
            set_frozen(model, freeze=True)
            logger.info("--- Phase 1: Head Warmup ---")
        elif epoch == 3:
            set_frozen(model, freeze=False)
            # Re-initialize optimizer state for full finetuning
            for g in optimizer.param_groups:
                g['lr'] = model_cfg["lr"]
            logger.info("--- Phase 2: Full Finetune ---")
        elif epoch == 16:
            for g in optimizer.param_groups:
                g["lr"] = 1e-6
            logger.info("--- Phase 3: Polish ---")

        # Train
        loss = train_one_epoch(
            epoch, model, ema_model, train_loader, optimizer, criterion, device,
            num_classes, plant_cfg, class_weights, phase, logger, scaler, use_amp
        )

        # Validate with EMA
        val_acc = validate(ema_model.module, val_loader, criterion, device, logger, plant_cfg)

        # Scheduler step (phases 1 & 2 only)
        if phase < 3:
            scheduler.step(epoch)

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/val",    val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Best model checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(
                ema_model.module.state_dict(),
                model_dir / f"best_{args.model}_{args.plant}.pth"
            )
            logger.info(f"New best: {best_acc:.2f}% -> saved best_{args.model}_{args.plant}.pth")
        else:
            patience += 1
            logger.info(f"No improvement. Patience {patience}/5")
            if phase == 2 and patience >= 10:
                logger.info("Early stopping triggered.")
                break

        # === Full epoch checkpoint (always — for crash recovery) ===
        save_checkpoint(
            {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "ema":       ema_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict() if scaler else {},
                "best_acc":  best_acc,
                "patience":  patience,
            },
            ckpt_dir,
        )

    logger.info(f"Training complete. Best Acc = {best_acc:.2f}%")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant",      type=str, required=True)
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train(args)
