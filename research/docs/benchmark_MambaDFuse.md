# MambaDFuse Benchmark — 16 Metrics

**Model:** MambaDFuse (checkpoint per dataset: `Medical_Fusion-{CT,PET,SPECT}-MRI/`)
**Datasets:** CT-MRI (24 imgs) · PET-MRI (24 imgs) · SPECT-MRI (24 imgs)
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)
**Eval script:**
```bash
python eval.py --model MambaDFuse --dataset CT-MRI
python eval.py --model MambaDFuse --dataset PET-MRI
python eval.py --model MambaDFuse --dataset SPECT-MRI
```

---

## CT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 71.3247 |
| SD | Basic Quality | 86.4515 |
| AG | Basic Quality | 8.0200 |
| EN | Basic Quality | 4.5995 |
| Q_MI | Info Theory | 0.7688 |
| Q_TE | Info Theory | 50.4443 |
| Q_NCIE | Info Theory | 0.0251 |
| Q_G | Img Feature | 0.5915 |
| Q_M | Img Feature | 0.6393 |
| Q_SF | Img Feature | -0.1497 |
| Q_P | Img Feature | 0.1397 |
| Q_S | Struct Sim | 0.3983 |
| Q_C | Struct Sim | 0.4489 |
| Q_Y | Struct Sim | 0.9266 |
| Q_CV | Perception | -1236.4122 |
| Q_CB | Perception | 0.6582 |

---

## PET-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 89.1021 |
| SD | Basic Quality | 86.6678 |
| AG | Basic Quality | 11.4913 |
| EN | Basic Quality | 6.3852 |
| Q_MI | Info Theory | 0.6593 |
| Q_TE | Info Theory | 21.3601 |
| Q_NCIE | Info Theory | 0.0382 |
| Q_G | Img Feature | 0.7466 |
| Q_M | Img Feature | 0.7040 |
| Q_SF | Img Feature | -0.0469 |
| Q_P | Img Feature | 0.3265 |
| Q_S | Struct Sim | 0.5333 |
| Q_C | Struct Sim | 0.5630 |
| Q_Y | Struct Sim | 0.6044 |
| Q_CV | Perception | -71.8988 |
| Q_CB | Perception | 0.3834 |

---

## SPECT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 60.5506 |
| SD | Basic Quality | 59.4432 |
| AG | Basic Quality | 6.1366 |
| EN | Basic Quality | 5.2540 |
| Q_MI | Info Theory | 0.6859 |
| Q_TE | Info Theory | 25.0505 |
| Q_NCIE | Info Theory | 0.0253 |
| Q_G | Img Feature | 0.5958 |
| Q_M | Img Feature | 0.6313 |
| Q_SF | Img Feature | -0.1139 |
| Q_P | Img Feature | 0.1770 |
| Q_S | Struct Sim | 0.4343 |
| Q_C | Struct Sim | 0.4539 |
| Q_Y | Struct Sim | 0.5166 |
| Q_CV | Perception | -185.7260 |
| Q_CB | Perception | 0.3052 |

---

## Quan sát

- **PET-MRI**: Mean và AG cao nhất trong 3 dataset → ảnh PET có màu sắc tươi (giá trị pixel cao), fusion giữ được nhiều cường độ
- **Q_Y thấp ở PET/SPECT** (0.60 / 0.52) — cấu trúc bị thay đổi nhiều do ảnh nguồn khác modality nhiều hơn CT
- **CT-MRI**: Q_Y = 0.9266 tốt nhất vì CT và MRI đều grayscale, ít méo cấu trúc hơn
- **Q_CB thấp ở PET/SPECT** → color preservation kém (model output grayscale, mất màu từ PET/SPECT)
