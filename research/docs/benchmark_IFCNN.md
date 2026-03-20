# IFCNN Benchmark — 16 Metrics

**Model:** IFCNN-MAX (`snapshots/IFCNN-MAX.pth`) — general purpose, phù hợp cho medical image fusion
**Datasets:** CT-MRI (24 imgs) · PET-MRI (24 imgs) · SPECT-MRI (24 imgs)
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)
**Eval script:**
```bash
python eval.py --model IFCNN --dataset CT-MRI
python eval.py --model IFCNN --dataset PET-MRI
python eval.py --model IFCNN --dataset SPECT-MRI
```

---

## CT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 60.2726 |
| SD | Basic Quality | 74.1002 |
| AG | Basic Quality | 8.2790 |
| EN | Basic Quality | 4.6045 |
| Q_MI | Info Theory | 0.7156 |
| Q_TE | Info Theory | 41.3664 |
| Q_NCIE | Info Theory | 0.0222 |
| Q_G | Img Feature | 0.5714 |
| Q_M | Img Feature | 0.6218 |
| Q_SF | Img Feature | -0.1246 |
| Q_P | Img Feature | 0.1181 |
| Q_S | Struct Sim | 0.3986 |
| Q_C | Struct Sim | 0.4308 |
| Q_Y | Struct Sim | 0.9019 |
| Q_CV | Perception | -1215.0521 |
| Q_CB | Perception | 0.6631 |

---

## PET-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 68.9944 |
| SD | Basic Quality | 80.2710 |
| AG | Basic Quality | 11.4324 |
| EN | Basic Quality | 5.3608 |
| Q_MI | Info Theory | 0.6007 |
| Q_TE | Info Theory | 9.1653 |
| Q_NCIE | Info Theory | 0.0222 |
| Q_G | Img Feature | 0.6674 |
| Q_M | Img Feature | 0.5681 |
| Q_SF | Img Feature | -0.0026 |
| Q_P | Img Feature | 0.2495 |
| Q_S | Struct Sim | 0.5763 |
| Q_C | Struct Sim | 0.5894 |
| Q_Y | Struct Sim | 0.9223 |
| Q_CV | Perception | -185.1804 |
| Q_CB | Perception | 0.6119 |

---

## SPECT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 43.5288 |
| SD | Basic Quality | 55.9559 |
| AG | Basic Quality | 6.9166 |
| EN | Basic Quality | 4.8978 |
| Q_MI | Info Theory | 0.6987 |
| Q_TE | Info Theory | 24.8982 |
| Q_NCIE | Info Theory | 0.0256 |
| Q_G | Img Feature | 0.6305 |
| Q_M | Img Feature | 0.6766 |
| Q_SF | Img Feature | -0.0165 |
| Q_P | Img Feature | 0.2642 |
| Q_S | Struct Sim | 0.5605 |
| Q_C | Struct Sim | 0.5825 |
| Q_Y | Struct Sim | 0.9445 |
| Q_CV | Perception | -110.5868 |
| Q_CB | Perception | 0.6544 |

---

## Quan sát

- **Q_Y đồng đều cao** ở cả 3 dataset (0.90 / 0.92 / 0.94) → IFCNN bảo toàn cấu trúc tốt dù không được train riêng cho medical fusion
- **SPECT-MRI**: Q_Y = 0.9445 cao nhất — tốt hơn cả MambaDFuse (0.5166) và SwinFusion (không có)
- **PET/SPECT Q_CB cao** (0.61 / 0.65) hơn MambaDFuse (0.38 / 0.31) → IFCNN bảo toàn màu tốt hơn
- **IFCNN là general-purpose** (không fine-tune trên medical) nhưng Q_Y cạnh tranh với model chuyên biệt
