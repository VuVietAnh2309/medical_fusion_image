# LRFNet Benchmark — 16 Metrics

**Model:** LRFNet (`models/model.pth`) — Real-time medical image fusion guided by detail information
**Designed for:** PET-MRI and SPECT-MRI (color source + grayscale MRI)
**Datasets:** CT-MRI (24 imgs) · PET-MRI (24 imgs) · SPECT-MRI (24 imgs)
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)

---

## CT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 53.6103 |
| SD | Basic Quality | 63.2473 |
| AG | Basic Quality | 6.4138 |
| EN | Basic Quality | 4.4853 |
| Q_MI | Info Theory | 0.7781 |
| Q_TE | Info Theory | 53.0782 |
| Q_NCIE | Info Theory | 0.0264 |
| Q_G | Img Feature | 0.4833 |
| Q_M | Img Feature | 0.6397 |
| Q_SF | Img Feature | -0.4030 |
| Q_P | Img Feature | 0.1904 |
| Q_S | Struct Sim | 0.3346 |
| Q_C | Struct Sim | 0.3974 |
| Q_Y | Struct Sim | 0.8944 |
| Q_CV | Perception | -3927.1094 |
| Q_CB | Perception | 0.6620 |

---

## PET-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 72.6165 |
| SD | Basic Quality | 82.3316 |
| AG | Basic Quality | 9.8447 |
| EN | Basic Quality | 5.3009 |
| Q_MI | Info Theory | 0.7272 |
| Q_TE | Info Theory | 18.3691 |
| Q_NCIE | Info Theory | 0.0334 |
| Q_G | Img Feature | 0.7031 |
| Q_M | Img Feature | 0.6150 |
| Q_SF | Img Feature | -0.1515 |
| Q_P | Img Feature | 0.2992 |
| Q_S | Struct Sim | 0.5217 |
| Q_C | Struct Sim | 0.5454 |
| Q_Y | Struct Sim | 0.9359 |
| Q_CV | Perception | -111.9197 |
| Q_CB | Perception | 0.6067 |

---

## SPECT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 45.2949 |
| SD | Basic Quality | 58.8301 |
| AG | Basic Quality | 5.9349 |
| EN | Basic Quality | 4.7126 |
| Q_MI | Info Theory | 0.8090 |
| Q_TE | Info Theory | 35.9244 |
| Q_NCIE | Info Theory | 0.0319 |
| Q_G | Img Feature | 0.6549 |
| Q_M | Img Feature | 0.7095 |
| Q_SF | Img Feature | -0.1435 |
| Q_P | Img Feature | 0.2904 |
| Q_S | Struct Sim | 0.4338 |
| Q_C | Struct Sim | 0.4677 |
| Q_Y | Struct Sim | 0.9540 |
| Q_CV | Perception | -61.1742 |
| Q_CB | Perception | 0.6498 |

---

## Quan sát

- **Q_Y cao nhất** trong số các model non-CDDFuse: SPECT = 0.9540 (cao hơn CDDFuse_MIF 0.990 nhưng tốt hơn IFCNN 0.9445)
- **Q_MI cao** ở SPECT (0.8090) — LRFNet giữ mutual information tốt trên dữ liệu medical
- **Q_CV ổn** ở PET/SPECT (-111, -61) — contrast perception tốt
- **PET-MRI Q_G = 0.7031** — gradient quality cao, so sánh được với CDDFuse_MIF (0.696)
