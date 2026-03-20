# Benchmark Report: Medical Image Fusion — PET-MRI

**Date:** 2026-03-12
**Dataset:** Harvard Public Medical Imaging Collection — PET-MRI
**Test set:** 24 images
**Evaluation metrics:** EN, MI, VIF, SCD, Q^AB/F, MS-SSIM, FMI, SF (all: higher is better)

---

## 1. Models Evaluated

### FusionMamba
- **Paper:** FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion with Mamba
- **Venue:** Visual Intelligence, Springer 2024 (peer-reviewed journal)
- **GitHub:** https://github.com/millieXie/FusionMamba
- **Architecture:** VMamba + Dynamic Feature Enhancement Module (DFEM) + Cross-Modal Fusion Mamba Module (CMFM)
- **Training:** Trained from scratch on Harvard PET-MRI train set (245 images)

### MambaDFuse
- **Paper:** MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion
- **Venue:** arXiv 2024 (preprint)
- **GitHub:** https://github.com/Lizhe1228/MambaDFuse
- **Architecture:** CNN + Mamba dual-phase fusion (shallow channel exchange + deep M3 blocks)
- **Training:** Used official pretrained weights (Medical_Fusion-PET-MRI, 10,000 iterations)

---

## 2. Benchmark Results

| Metric | FusionMamba (paper) | FusionMamba (50 ep) | FusionMamba (30k steps, aug) | MambaDFuse (pretrained) |
|--------|:-------------------:|:-------------------:|:---------------------------:|:-----------------------:|
| EN | - | 4.9495 | 5.3345 | **6.3852** |
| MI | - | 2.1477 | 3.0521 | **3.4245** |
| VIF | **0.6920** | 0.1969 | 0.3108 | 0.3804 |
| SCD | 1.3906 | 1.3091 | 1.4577 | **1.5317** |
| Q^AB/F | **0.7406** | 0.3478 | 0.5767 | 0.7051 |
| MS-SSIM | **0.9362** | 0.6775 | 0.6889 | 0.6886 |
| FMI | **0.8616** | 0.2004 | 0.2543 | 0.3093 |
| SF | - | **53.6542** | 38.1516 | 34.8067 |

> Bold = best result per metric

---

## 3. Training Configuration

### FusionMamba
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.0001 (decay 0.75/epoch) |
| Batch size | 2 |
| Image size | 256×256 |
| Epochs (exp 1) | 2 |
| Epochs (exp 2) | 50 |
| Exp 3 | 30,000 steps (2 epochs × 15,000 batches), augmentation, MSE weight ×100 |
| Train samples | 245 (augmented to 30,000 effective) |
| Environment | Python 3.8, PyTorch 1.13, CUDA 11.7 |

### MambaDFuse
| Parameter | Value |
|-----------|-------|
| Pretrained iterations | 10,000 |
| Image size | 128×128 (window_size=8) |
| Environment | Python 3.8, PyTorch 1.13, CUDA 11.7 |

---

## 4. Analysis

### FusionMamba (50 ep, no aug) → (30k steps, aug): Improvement
| Metric | Δ | % Change |
|--------|---|---------|
| EN | +0.385 | +7.8% |
| MI | +0.904 | +42.1% |
| VIF | +0.114 | +57.8% |
| SCD | +0.149 | +11.4% |
| Q^AB/F | +0.229 | +65.8% |
| MS-SSIM | +0.011 | +1.7% |
| FMI | +0.054 | +26.9% |
| SF | -15.503 | -28.9% |

> Augmentation (random flip, rot90, crop 288→256) + loss weight sửa theo paper (MSE ×100) giúp cải thiện rất lớn.
> MS-SSIM vượt MambaDFuse (0.6889 vs 0.6886).

### FusionMamba (30k steps, aug) vs MambaDFuse
- MambaDFuse vẫn thắng 6/8 metrics (EN, MI, VIF, SCD, Q^AB/F, FMI)
- FusionMamba thắng **MS-SSIM** (0.6889 vs 0.6886) và **SF** (38.15 vs 34.81)
- Khoảng cách đã thu hẹp đáng kể so với trước augmentation

---

## 5. Notes & Next Steps

- [x] Data augmentation (random flip, rot90, crop) + loss weight theo paper → cải thiện lớn
- [ ] Liên hệ tác giả FusionMamba xin pretrained PET-MRI weights (link hết hạn)
- [ ] Thử CDDFuse (CVPR 2023) và EMMA (CVPR 2024) để có thêm baseline
- [ ] Tăng epochs (4–10) với augmentation để tiếp tục cải thiện

---

## 6. Environment

```
GPU: NVIDIA GeForce RTX 4090 (24GB)
CUDA Driver: 580.126.09 (CUDA 13.0)
Python: 3.8
PyTorch: 1.13.0+cu117
mamba_ssm: 1.0.1
causal_conv1d: 1.0.0
Conda env: mri
```

## 7. Paths

```
Dataset:      /data2/anhvv/ai_nlp_rs/datasets/data_PET-MRI/
FusionMamba:  /data2/anhvv/ai_nlp_rs/FusionMamba/
MambaDFuse:   /data2/anhvv/ai_nlp_rs/MambaDFuse/
Outputs:
  FusionMamba: /data2/anhvv/ai_nlp_rs/FusionMamba/outputs/PET-MRI/
  MambaDFuse:  /data2/anhvv/ai_nlp_rs/MambaDFuse/results/MambaDFuse_PET-MRI/
```
