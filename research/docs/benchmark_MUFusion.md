# MUFusion Benchmark — 16 Metrics

**Model:** MUFusion (medical) — `harvard.model`
**Checkpoint:** Trained on Harvard Medical Atlas dataset
**Datasets:** CT-MRI (24 imgs) · PET-MRI (24 imgs) · SPECT-MRI (24 imgs)
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)

---

## CT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 60.6874 |
| SD | Basic Quality | 68.4810 |
| AG | Basic Quality | 6.4910 |
| EN | Basic Quality | 5.3012 |
| Q_MI | Info Theory | 0.6037 |
| Q_TE | Info Theory | 36.7468 |
| Q_NCIE | Info Theory | 0.0196 |
| Q_G | Img Feature | 0.4289 |
| Q_M | Img Feature | 0.5885 |
| Q_SF | Img Feature | -0.4398 |
| Q_P | Img Feature | 0.0829 |
| Q_S | Struct Sim | 0.3431 |
| Q_C | Struct Sim | 0.3868 |
| Q_Y | Struct Sim | 0.6793 |
| Q_CV | Perception | -1948.9952 |
| Q_CB | Perception | 0.3583 |

---

## PET-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 76.8390 |
| SD | Basic Quality | 80.2252 |
| AG | Basic Quality | 9.9527 |
| EN | Basic Quality | 6.0227 |
| Q_MI | Info Theory | 0.5711 |
| Q_TE | Info Theory | 10.0459 |
| Q_NCIE | Info Theory | 0.0236 |
| Q_G | Img Feature | 0.6510 |
| Q_M | Img Feature | 0.5521 |
| Q_SF | Img Feature | -0.1793 |
| Q_P | Img Feature | 0.1984 |
| Q_S | Struct Sim | 0.5358 |
| Q_C | Struct Sim | 0.5522 |
| Q_Y | Struct Sim | 0.7787 |
| Q_CV | Perception | -178.0523 |
| Q_CB | Perception | 0.4049 |

---

## SPECT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 49.8444 |
| SD | Basic Quality | 59.7115 |
| AG | Basic Quality | 5.9499 |
| EN | Basic Quality | 5.2669 |
| Q_MI | Info Theory | 0.6914 |
| Q_TE | Info Theory | 24.1016 |
| Q_NCIE | Info Theory | 0.0277 |
| Q_G | Img Feature | 0.5861 |
| Q_M | Img Feature | 0.6381 |
| Q_SF | Img Feature | -0.1830 |
| Q_P | Img Feature | 0.2167 |
| Q_S | Struct Sim | 0.4817 |
| Q_C | Struct Sim | 0.5148 |
| Q_Y | Struct Sim | 0.7870 |
| Q_CV | Perception | -131.6145 |
| Q_CB | Perception | 0.3868 |

---

## Quan sát

- **EN cao** (5.30/6.02/5.27) — entropy lớn, ảnh fused chứa nhiều thông tin
- **Q_Y trung bình** (~0.68/0.78/0.79) — thấp hơn LRFNet và IFCNN
- **Q_SF âm** ở cả 3 dataset → spatial frequency bị giảm sau fusion (ảnh có xu hướng mịn hơn source)
- **Q_CV ổn** ở PET/SPECT (-178, -131) nhưng kém ở CT-MRI (-1949)
