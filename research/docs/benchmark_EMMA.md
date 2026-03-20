# EMMA Benchmark — 16 Metrics

**Model:** EMMA (CVPR 2024) — Equivariant Multi-Modality Image Fusion
**Checkpoint:** `model/EMMA.pth`
**Datasets:** CT-MRI (24 imgs) · PET-MRI (24 imgs) · SPECT-MRI (24 imgs)
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)

---

## CT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 68.4237 |
| SD | Basic Quality | 79.4804 |
| AG | Basic Quality | 7.3732 |
| EN | Basic Quality | 5.3642 |
| Q_MI | Info Theory | 0.6821 |
| Q_TE | Info Theory | 50.3070 |
| Q_NCIE | Info Theory | 0.0242 |
| Q_G | Img Feature | 0.5099 |
| Q_M | Img Feature | 0.5814 |
| Q_SF | Img Feature | -0.3315 |
| Q_P | Img Feature | 0.0963 |
| Q_S | Struct Sim | 0.3716 |
| Q_C | Struct Sim | 0.4162 |
| Q_Y | Struct Sim | 0.8853 |
| Q_CV | Perception | -1597.8158 |
| Q_CB | Perception | 0.5507 |

---

## PET-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 82.6768 |
| SD | Basic Quality | 89.6352 |
| AG | Basic Quality | 10.7801 |
| EN | Basic Quality | 6.2169 |
| Q_MI | Info Theory | 0.5930 |
| Q_TE | Info Theory | 11.0842 |
| Q_NCIE | Info Theory | 0.0272 |
| Q_G | Img Feature | 0.6666 |
| Q_M | Img Feature | 0.5370 |
| Q_SF | Img Feature | -0.1225 |
| Q_P | Img Feature | 0.1780 |
| Q_S | Struct Sim | 0.5248 |
| Q_C | Struct Sim | 0.5469 |
| Q_Y | Struct Sim | 0.9080 |
| Q_CV | Perception | -106.5896 |
| Q_CB | Perception | 0.4977 |

---

## SPECT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 52.1951 |
| SD | Basic Quality | 65.1840 |
| AG | Basic Quality | 7.0186 |
| EN | Basic Quality | 5.6084 |
| Q_MI | Info Theory | 0.6934 |
| Q_TE | Info Theory | 28.7539 |
| Q_NCIE | Info Theory | 0.0295 |
| Q_G | Img Feature | 0.6276 |
| Q_M | Img Feature | 0.6171 |
| Q_SF | Img Feature | -0.0334 |
| Q_P | Img Feature | 0.1572 |
| Q_S | Struct Sim | 0.4597 |
| Q_C | Struct Sim | 0.4930 |
| Q_Y | Struct Sim | 0.9193 |
| Q_CV | Perception | -87.0556 |
| Q_CB | Perception | 0.5325 |

---

## Quan sát

- **Q_Y cao nhất** trong nhóm non-specialized: SPECT = 0.9193 (gần bằng LRFNet 0.9540)
- **PET-MRI Mean = 82.68, SD = 89.64** — nổi bật về basic quality
- **Q_CV ổn ở PET/SPECT** (-106, -87) — perception contrast tốt hơn nhiều model khác
- **SPECT Q_SF = -0.034** — rất gần 0, gần như không mất spatial frequency
