# NestFuse Benchmark — 16 Metrics

**Model:** NestFuse (`models/nestfuse_1e2.model`) — attention_avg fusion strategy
**Datasets:** CT-MRI (24 imgs) · PET-MRI (24 imgs) · SPECT-MRI (24 imgs)
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)

---

## CT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 52.1070 |
| SD | Basic Quality | 65.1626 |
| AG | Basic Quality | 7.1338 |
| EN | Basic Quality | 4.3418 |
| Q_MI | Info Theory | 0.7236 |
| Q_TE | Info Theory | 33.0615 |
| Q_NCIE | Info Theory | 0.0214 |
| Q_G | Img Feature | 0.3687 |
| Q_M | Img Feature | 0.6177 |
| Q_SF | Img Feature | -0.2834 |
| Q_P | Img Feature | 0.1003 |
| Q_S | Struct Sim | 0.2543 |
| Q_C | Struct Sim | 0.2931 |
| Q_Y | Struct Sim | 0.8144 |
| Q_CV | Perception | -3497.4759 |
| Q_CB | Perception | 0.6546 |

---

## PET-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 58.1274 |
| SD | Basic Quality | 86.9134 |
| AG | Basic Quality | 5.8675 |
| EN | Basic Quality | 4.0585 |
| Q_MI | Info Theory | 0.7458 |
| Q_TE | Info Theory | 13.5632 |
| Q_NCIE | Info Theory | 0.0215 |
| Q_G | Img Feature | 0.2334 |
| Q_M | Img Feature | 0.5136 |
| Q_SF | Img Feature | -0.3900 |
| Q_P | Img Feature | 0.0969 |
| Q_S | Struct Sim | 0.2239 |
| Q_C | Struct Sim | 0.2442 |
| Q_Y | Struct Sim | 0.6824 |
| Q_CV | Perception | -2607.8809 |
| Q_CB | Perception | 0.5674 |

---

## SPECT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 41.2956 |
| SD | Basic Quality | 65.4226 |
| AG | Basic Quality | 5.4052 |
| EN | Basic Quality | 3.6771 |
| Q_MI | Info Theory | 0.6553 |
| Q_TE | Info Theory | 17.8020 |
| Q_NCIE | Info Theory | 0.0160 |
| Q_G | Img Feature | 0.3445 |
| Q_M | Img Feature | 0.5834 |
| Q_SF | Img Feature | -0.0819 |
| Q_P | Img Feature | 0.0839 |
| Q_S | Struct Sim | 0.2491 |
| Q_C | Struct Sim | 0.2591 |
| Q_Y | Struct Sim | 0.7913 |
| Q_CV | Perception | -854.5672 |
| Q_CB | Perception | 0.6108 |

---

## Quan sát

- **Q_Y trung bình** (~0.81/0.68/0.79) — thấp hơn IFCNN và CDDFuse, do NestFuse không được train trên medical data
- **Q_CV rất thấp** (đặc biệt CT-MRI: -3497) → contrast giữa fused và source kém
- **Q_G/Q_S/Q_C thấp** — cấu trúc gradient và structural similarity thấp hơn các model chuyên biệt
