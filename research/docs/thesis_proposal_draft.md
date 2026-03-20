# Thesis Proposal Draft: Enhancing Multimodal Medical Image Fusion via Frequency-Aware Decomposition

## 1. Motivation

Từ benchmark 11 models trên 3 datasets (CT-MRI, PET-MRI, SPECT-MRI), ta rút ra các quan sát:

**CDDFuse_MIF thắng áp đảo** (best trên 10-12/18 metrics mỗi dataset) nhờ sử dụng **dual-branch decomposition** — tách ảnh thành base layer (LF) và detail layer (HF), xử lý riêng rồi merge.

Tuy nhiên, CDDFuse vẫn có **điểm yếu rõ ràng:**

| Điểm yếu | Metrics thể hiện | Nguyên nhân |
|-----------|-----------------|-------------|
| SCD thấp hơn CDDFuse_IVF | SCD: 1.40 vs 1.74 (CT-MRI) | LF branch (CNN đơn giản) không capture đủ global context → mất cân bằng modality |
| Q_S, Q_C thấp | Q_S: 0.44, Q_C: 0.49 | Structural similarity cục bộ chưa tốt — fusion chưa adaptive theo vùng |
| Q_CV rất âm | Q_CV: −1028 (CT-MRI) | Perceptual distortion cao tại vùng salient — HF reconstruction chưa chính xác |
| EN thấp | EN: 4.77 (CT-MRI) | Ảnh fusion nghèo thông tin hơn một số model khác (EMMA: 5.36) |

**Các models khác cũng không giải quyết được toàn diện:**
- MambaDFuse: VIF cao nhưng SCD, Q_Y thấp (thiên lệch modality)
- DDFM (diffusion): Q_MI, Q_TE cao nhưng Q_G, Q_SF rất thấp (mất cạnh)
- IFCNN: Q_SF tốt nhất nhưng VIF, SCD thấp (mất thông tin thị giác)

→ **Không model nào đạt best trên tất cả metrics.** Cơ hội cải tiến nằm ở việc **kết hợp điểm mạnh** của nhiều approach.

---

## 2. Ý tưởng chính

### Pipeline đề xuất: Frequency-Aware Dual-Branch Fusion

```
Input A (CT/PET/SPECT) -─┐
                         ├── Frequency Decomposition ──┬── HF Branch ──┐
Input B (MRI) ───────────┘                             |               ├── Merge ── F
                                                       └── LF Branch ──┘
```

**Bước 1: Frequency Decomposition (Tách HF/LF)**

Tách mỗi ảnh nguồn thành 2 thành phần:
- **HF (High Frequency):** cạnh, texture, chi tiết mịn — thông tin giải phẫu (edges of bones, tissue boundaries)
- **LF (Low Frequency):** cấu trúc tổng thể, intensity distribution — thông tin chức năng (organ shapes, brightness patterns)

Các phương án decomposition:

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| **NSST** (Non-Subsampled Shearlet Transform) | Shift-invariant, multi-direction (8-16 hướng), bảo toàn cạnh tốt | Cố định, không learnable |
| **INN** (Invertible Neural Network, CDDFuse dùng) | Learnable, end-to-end | Decomposition không rõ ràng về tần số |
| **Learnable Frequency Filter** (đề xuất mới) | Adaptive theo dataset, interpretable | Cần thiết kế cẩn thận |
| **Octave Convolution** | Efficient, giảm redundancy | Chỉ 2 mức (high/low), không multi-scale |

**Đề xuất:** Kết hợp **NSST decomposition** (đảm bảo tách tần số rõ ràng, có cơ sở toán học) + **learnable refinement** (CNN nhỏ để tinh chỉnh HF/LF components sau khi tách).

---

**Bước 2: HF Branch — Edge & Detail Fusion**

HF chứa cạnh và texture — cần model có khả năng bảo toàn cạnh tốt.

Các phương án:

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| **SML max selection** (truyền thống) | Đơn giản, interpretable | Không adaptive, artifact tại boundary |
| **CNN + Attention** | Learnable, local detail | Thiếu global context |
| **Transformer** (CDDFuse dùng) | Global attention, cross-modal | Quadratic complexity O(n²) |
| **Mamba/SSM** (FusionMamba dùng) | Linear complexity O(n), long-range | Yếu local detail |
| **CNN-Mamba hybrid** (đề xuất) | Local + global, efficient | Cần thiết kế cẩn thận |

**Đề xuất:** Sử dụng **CNN blocks cho local edge refinement** + **Mamba/SSM cho cross-modal long-range dependency**. CNN giữ local edges sharp, Mamba đảm bảo consistency giữa 2 modality ở scale lớn.

Cụ thể:
```
HF_A, HF_B ──→ [Local CNN] ──→ [Cross-Modal Mamba] ──→ [Edge Attention] ──→ HF_fused
```

- **Local CNN:** 3-5 layers conv, bảo toàn gradient magnitude và orientation
- **Cross-Modal Mamba:** FusionMamba-style CMFM, scan HF features từ cả 2 modality
- **Edge Attention:** Sobel-guided attention map, tăng trọng số tại vùng cạnh mạnh

**Kỳ vọng cải thiện:** Q^{AB/F} ↑, Q_P ↑, FMI ↑, Q_SF ↑ (bảo toàn cạnh tốt hơn)

---

**Bước 3: LF Branch — Structure & Intensity Fusion**

LF chứa cấu trúc tổng thể và phân bố intensity — cần model capture global context và cân bằng 2 modality.

Các phương án:

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| **Average** (truyền thống) | Đơn giản | Mất contrast, over-smoothing |
| **Weighted average** (variance-based) | Adaptive | Vẫn linear, limited |
| **CNN** (CDDFuse dùng) | Learnable | Thiếu global context |
| **Mamba/SSM** (đề xuất) | Global context, linear O(n) | Yếu local |
| **Mamba + Modality Balancing** (đề xuất) | Global + balanced | Cần thiết kế loss |

**Đề xuất:** Sử dụng **Mamba/SSM** cho LF branch vì LF cần global context (toàn bộ cấu trúc ảnh). Thêm **Modality Balancing Module** để đảm bảo cả 2 nguồn đóng góp đều:

```
LF_A, LF_B ──→ [Mamba Encoder] ──→ [Modality Balance] ──→ [Mamba Decoder] ──→ LF_fused
```

- **Mamba Encoder:** Capture long-range spatial dependencies trong LF
- **Modality Balance:** Tính saliency map cho mỗi nguồn, normalize contribution
- **Mamba Decoder:** Reconstruct balanced LF fused component

**Kỳ vọng cải thiện:** SCD ↑, Q_S ↑, Q_C ↑, VIF ↑ (cân bằng modality, bảo toàn structure)

---

**Bước 4: Merge — Frequency Reconstruction**

Merge HF_fused và LF_fused thành ảnh F cuối cùng.

Các phương án:

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| **Inverse NSST** (nếu dùng NSST decomposition) | Lossless, exact reconstruction | Cố định |
| **Learned reconstruction** (CNN decoder) | Adaptive | Có thể thêm artifact |
| **Residual addition** (đề xuất) | Interpretable: F = LF_fused + HF_fused | Cần matching scale |

**Đề xuất:** Nếu dùng NSST decomposition → dùng **Inverse NSST** để đảm bảo reconstruction chính xác. Thêm **lightweight refinement CNN** (2-3 layers) sau inverse NSST để giảm artifact.

---

**Bước 5: Loss Function — Multi-Objective**

```
L_total = α₁ × L_int + α₂ × L_texture + α₃ × L_ssim + α₄ × L_perceptual
```

| Loss | Mục đích | Target metrics |
|------|---------|---------------|
| L_int = ‖F − max(A,B)‖₁ | Bảo toàn intensity | VIF ↑, SCD ↑ |
| L_texture = ‖∇F − max(∇A, ∇B)‖₁ | Bảo toàn cạnh/gradient | Q^{AB/F} ↑, Q_SF ↑ |
| L_ssim = 1 − ½[SSIM(A,F) + SSIM(B,F)] | Bảo toàn cấu trúc | MS-SSIM ↑, Q_Y ↑ |
| L_perceptual = CSF-weighted distortion | Giảm perceptual distortion | Q_CV ↑, Q_CB ↑ |

**Điểm mới:** L_perceptual dựa trên CSF (Contrast Sensitivity Function) — cùng nguyên lý với Q_CV metric, nhưng dùng làm loss để train. Điều này trực tiếp tối ưu hóa metric Q_CV mà các model hiện tại đều yếu.

---

## 3. Novelty (Điểm mới so với existing work)

| Existing | Limitation | Proposed Improvement |
|----------|-----------|---------------------|
| CDDFuse: INN decomposition | Decomposition không rõ ràng về tần số, LF branch quá đơn giản (CNN) | NSST decomposition (tần số rõ ràng) + Mamba cho LF (global context) |
| FusionMamba: implicit decomposition | Không tách HF/LF explicitly, khó interpretable | Explicit HF/LF + CNN-Mamba hybrid cho HF |
| MFDF+NSST: manual rules | Max selection/averaging không adaptive | Learned fusion rules (CNN+Mamba) thay manual rules |
| DDFM: diffusion-based | Rất chậm (50-1000 steps), mất cạnh | Real-time inference, edge-preserving HF branch |

**Tóm tắt novelty:**
1. **Explicit frequency decomposition + Learned fusion:** Kết hợp NSST (interpretable, có cơ sở toán) với deep learning (adaptive, learned rules)
2. **Asymmetric dual-branch:** HF dùng CNN-Mamba (local+global edge), LF dùng pure Mamba (global structure) — mỗi branch tối ưu cho đặc tính tần số riêng
3. **Perceptual loss (CSF-based):** Trực tiếp tối ưu Q_CV — metric mà tất cả 11 model đều yếu
4. **Modality Balancing Module:** Đảm bảo cân bằng A và B trong LF branch → cải thiện SCD

---

## 4. Expected Results

Dựa trên phân tích benchmark, target cải thiện:

| Metric | CDDFuse_MIF (baseline) | Target | Cải thiện bởi |
|--------|----------------------|--------|--------------|
| MS-SSIM | 0.643 | 0.66+ | L_ssim + LF Mamba |
| FMI | 0.356 | 0.37+ | HF CNN-Mamba |
| VIF | 0.303 | 0.32+ | LF Modality Balance |
| SCD | 1.400 | 1.65+ | LF Modality Balance |
| Q^{AB/F} | 0.675 | 0.70+ | HF Edge Attention |
| Q_CV | −1028 | −800+ | L_perceptual (CSF) |
| Q_CB | 0.702 | 0.72+ | L_perceptual (CSF) |
| EN | 4.77 | 5.0+ | Balanced fusion |

---

## 5. Implementation Plan

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Implement NSST decomposition + inverse | 2 weeks |
| 2 | Build HF branch (CNN + Mamba hybrid) | 3 weeks |
| 3 | Build LF branch (Mamba + Modality Balance) | 3 weeks |
| 4 | Merge module + Loss function design | 1 week |
| 5 | Training on Harvard medical dataset | 2 weeks |
| 6 | Evaluation (18 metrics × 3 datasets) + Analysis | 1 week |
| 7 | Ablation study + Paper writing | 2 weeks |

**Dataset:** Harvard public medical dataset (CT-MRI: 166 pairs, PET-MRI: 329 pairs, SPECT-MRI: 539 pairs), augmented to 30000 pairs via rotation.

**Hardware:** 2× RTX 4090

---

## 6. References

- Xie et al. (2024) — FusionMamba: SSM for image fusion
- Zhao et al. (2023) — CDDFuse: Dual-branch decomposition (CVPR)
- Liu et al. (2012) — Benchmark 12 metrics (IEEE TPAMI)
- Wang & Bovik (2004) — SSIM (IEEE TIP)
- MFDF+NSST (2021) — NSST-based medical fusion
- Hermessi et al. (2021) — NSST review for medical fusion
