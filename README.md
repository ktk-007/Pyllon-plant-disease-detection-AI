# 🌿 Pyllon

> **Per-plant ensemble deep learning for crop disease classification** — pairing ConvNeXt-Tiny & EfficientNetV2-S per species, with three-phase transfer learning, EMA, TTA, and dynamic quantisation. Deployed as a Streamlit web app with an integrated Claude Haiku conversational assistant.
> **Streamlit liveApp** -- https://pyllon-plant-disease-detection-ai-conveffenet.streamlit.app/
---

## Overview

Pyllon (from Greek *phyllon*, leaf) addresses a core limitation of conventional plant disease models: training a single classifier across all plants simultaneously forces the network to learn inter-species variation, which introduces confounding features. Pyllon instead trains **one dedicated ensemble pair per plant**, restricting each model's decision space to intra-plant disease differentiation only.

The system covers **10 sub-models across 9 crop species** (Mango is split into Leaf and Fruit), with **40 disease categories** in total. Each ensemble unit pairs:

- **ConvNeXt-Tiny** — large 7×7 depthwise convolutions, excellent for spatially extended disease patterns (rings, diffuse mildew)
- **EfficientNetV2-S** — fine-grained feature sharpness, strong on lesion boundary detail and colour gradient transitions

Final ensemble prediction:

```
P_final = 0.55 × P_ConvNeXt + 0.45 × P_EfficientNet
```

---

## Model Specifications

| Property | ConvNeXt-Tiny | EfficientNetV2-S |
|---|---|---|
| Parameters | ~28 million | ~22 million |
| Input resolution | 224 × 224 | 224 × 224 |
| Feature dim | 768 | 1280 |
| Pretrained on | ImageNet-1K | ImageNet-1K |
| Quantised size | ~28 MB | ~22 MB |
| timm identifier | `convnext_tiny` | `tf_efficientnetv2_s` |

---

## Plant & Disease Coverage

| Plant / Sub-model | Classes | Loss | Disease Classes |
|---|---|---|---|
| Tomato | 6 | CrossEntropy | bacterial_spot, early_blight, healthy, late_blight, septoria_leaf_spot, yellow_leaf_curl_virus |
| Mango Leaf | 7 | CrossEntropy | anthracnose, bacterial_canker, black_mould_rot, gall_midge, powdery_mildew, sooty_mould, healthy |
| Mango Fruit | 4 | CrossEntropy | anthracnose, black_mould_rot, stem_end_rot, healthy |
| Apple | 4 | CrossEntropy | black_rot, healthy, rust, scab |
| Potato | 3 | CrossEntropy | early_blight, healthy, late_blight |
| Rose | 5 | **Focal Loss** | black_spot, downy_mildew, healthy, powdery_mildew, rust |
| Corn | 4 | CrossEntropy | gray_leaf_spot, healthy, northern_leaf_blight, rust |
| Bell Pepper | 2 | CrossEntropy | bacterial_spot, healthy |
| Grape | 4 | CrossEntropy | black_rot, esca, healthy, leaf_blight |
| Strawberry | 2 | CrossEntropy | healthy, leaf_scorch |

---

## Training Methodology

### Three-Phase Protocol

| Phase | Epochs | Backbone | Augmentation | Notes |
|---|---|---|---|---|
| 1 — Warmup | 0–4 | Frozen | Standard | LR ramps linearly to peak |
| 2 — Finetune | 5–29 | Unfrozen | MixUp + Label Smoothing | Cosine LR decay |
| 3 — Polish | 30–34 | Unfrozen | Standard only | Fixed LR 1×10⁻⁶, EMA finalised |

### Augmentation Pipeline (Albumentations)

- `RandomResizedCrop(224×224, scale=0.7–1.0)` — forces focus on leaf tissue regardless of background
- `HorizontalFlip` / `VerticalFlip` (p=0.5 each) — disease patterns are orientation-invariant
- `RandomBrightnessContrast` (±0.3, p=0.5) — handles variable field lighting
- `HueSaturationValue`, `GaussianBlur`, `GridDistortion` — colour and texture robustness
- Validation / test: `Resize(256) → CenterCrop(224) → Normalize` only

### Optimiser Hyperparameters

| Parameter | ConvNeXt-Tiny | EfficientNetV2-S |
|---|---|---|
| Peak LR | 1×10⁻³ | 5×10⁻⁴ |
| Weight Decay | 0.05 | 0.01 |
| Betas | (0.9, 0.999) | (0.9, 0.999) |
| Warmup LR | 1×10⁻⁶ | 1×10⁻⁶ |

Adaptive checkpoint strategy: save on validation accuracy improvement; halt training if loss diverges beyond an acceptable threshold.

---

## Inference Pipeline

### Test Time Augmentation (TTA)

Each image is processed through 5 geometric variants; softmax outputs are averaged before ensemble combination:

| Pass | Transform |
|---|---|
| 1 | Original (centre crop) |
| 2 | Horizontal flip |
| 3 | Vertical flip |
| 4 | 90° rotation |
| 5 | RandomResizedCrop (scale=0.9) |

For multi-image uploads (1–5 images), TTA averaging is performed per image, then probability vectors are averaged across images before ensemble combination.

### Model Quantisation & RAM Budget

| Model | Unquantised | Quantised |
|---|---|---|
| ConvNeXt-Tiny | ~110 MB | ~28 MB |
| EfficientNetV2-S | ~88 MB | ~22 MB |
| Pair (1 plant) | ~198 MB | ~50 MB |
| Peak RAM (deployed) | ~198 MB | **~50 MB** (1 pair active) |

Dynamic quantisation (`torch.quantization.quantize_dynamic`) reduces each model pair to ~50 MB, enabling full deployment within Streamlit's **1 GB RAM constraint**.

---

## Dataset

| Split | Images/Class | Purpose |
|---|---|---|
| Train | ~2,000 (Rose: ~1,000) | Model learning |
| Val | 240–460 | Hyperparameter tuning & checkpoint selection |
| Test | ~50 | Final evaluation only — never seen during training |

**Mango specialist preprocessing:** images are encoded with a three-prefix naming convention (`MAN_LF_` leaf, `MAN_BG_` background-removed, `MAN_FR_` fruit) to handle the structurally heterogeneous dataset.

---

## Expected Performance

| Plant / Sub-model | Classes | Single Model | Ensemble + EMA + TTA |
|---|---|---|---|
| Tomato | 6 | 93–96% | **95–97%** |
| Mango Leaf | 7 | 88–91% | **91–94%** |
| Mango Fruit | 4 | 89–92% | **92–95%** |
| Apple | 4 | 94–97% | **96–98%** |
| Potato | 3 | 96–98% | **97–99%** |
| Rose | 5 | 86–89% | **89–93%** |
| Corn | 4 | 94–97% | **96–98%** |
| Bell Pepper | 2 | 97–98% | **98–99%** |
| Grape | 4 | 92–94% | **94–97%** |
| Strawberry | 2 | 96–97% | **97–99%** |

> Achieving high validation accuracy under simultaneous MixUp and Label Smoothing represents a qualitatively stronger result than the same accuracy without regularisers — these techniques actively prevent memorisation, so accuracy reflects genuine feature understanding.

---

## Streamlit Application

The web app is dark-themed with the following workflow:

1. User selects plant from sidebar dropdown
2. For Mango, a secondary leaf/fruit selector appears
3. User uploads 1–5 images
4. Pyllon runs TTA → ensemble → returns predicted class + confidence
5. A scoped **Claude Haiku**-powered conversational assistant appears below results for farmer-facing disease guidance (max 150 tokens, structured system prompt scoped to the predicted disease only)

---

## Tech Stack

| Component | Library |
|---|---|
| Model training | PyTorch, timm |
| Augmentation | Albumentations |
| Quantisation | `torch.quantization.quantize_dynamic` |
| Web app | Streamlit |
| Conversational AI | Anthropic Claude Haiku |
| Model hosting | Streamlit resource caching |

---

## Project Name

*Pyllon* is derived from the Greek word **φύλλον** (*phyllon*), meaning leaf — adapted to Python naming conventions. The name reflects both the botanical focus of the system and its Python-native implementation.
