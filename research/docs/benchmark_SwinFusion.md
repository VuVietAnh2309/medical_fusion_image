# SwinFusion Benchmark — 16 Metrics

**Model:** SwinFusion
**Checkpoint:** `Medical_Fusion-CT-MRI/` · `Medical_Fusion-PET-MRI/` · SPECT-MRI: không có
**Datasets:** CT-MRI (24 imgs) · PET-MRI (24 imgs) · SPECT-MRI: ---
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)
**Eval script:**
```bash
python eval.py --model SwinFusion --dataset CT-MRI
python eval.py --model SwinFusion --dataset PET-MRI
```

---

## CT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 69.1597 |
| SD | Basic Quality | 85.2622 |
| AG | Basic Quality | 7.3459 |
| EN | Basic Quality | 4.5382 |
| Q_MI | Info Theory | 0.7896 |
| Q_TE | Info Theory | 56.7183 |
| Q_NCIE | Info Theory | 0.0259 |
| Q_G | Img Feature | 0.5553 |
| Q_M | Img Feature | 0.6434 |
| Q_SF | Img Feature | -0.2087 |
| Q_P | Img Feature | 0.1277 |
| Q_S | Struct Sim | 0.3834 |
| Q_C | Struct Sim | 0.4369 |
| Q_Y | Struct Sim | 0.9336 |
| Q_CV | Perception | -1211.6005 |
| Q_CB | Perception | 0.6558 |

---

## PET-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 82.8005 |
| SD | Basic Quality | 78.5497 |
| AG | Basic Quality | 10.9130 |
| EN | Basic Quality | 5.9904 |
| Q_MI | Info Theory | 0.5354 |
| Q_TE | Info Theory | 8.0032 |
| Q_NCIE | Info Theory | 0.0193 |
| Q_G | Img Feature | 0.6668 |
| Q_M | Img Feature | 0.5276 |
| Q_SF | Img Feature | -0.1560 |
| Q_P | Img Feature | 0.1763 |
| Q_S | Struct Sim | 0.5121 |
| Q_C | Struct Sim | 0.5242 |
| Q_Y | Struct Sim | 0.5607 |
| Q_CV | Perception | -434.8382 |
| Q_CB | Perception | 0.3631 |

---

## SPECT-MRI

| Metric | Group | Score |
|---|---|---|
| *tất cả* | — | --- |

> Không có checkpoint cho SPECT-MRI.

---

## Quan sát

- **CT-MRI**: Q_Y = 0.9336 — cấu trúc tốt, tương đương MambaDFuse (0.9266)
- **PET-MRI**: Q_Y = 0.5607, Q_MI = 0.5354 — thấp hơn MambaDFuse, do SwinFusion được train trên PET-MRI nhưng cross-modal gap lớn hơn
- **Q_SF âm** ở cả 2 dataset (đặc biệt CT-MRI: -0.2087) → spatial frequency của fusion thấp hơn nguồn, ảnh bị smooth
