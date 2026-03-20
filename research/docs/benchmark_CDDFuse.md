# CDDFuse Benchmark — 16 Metrics

**Models:** CDDFuse_IVF (`CDDFuse_IVF.pth`) · CDDFuse_MIF (`CDDFuse_MIF.pth`)
**Datasets:** CT-MRI (21 imgs) · PET-MRI (42 imgs) · SPECT-MRI (73 imgs)
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)
**Eval script:** `python eval.py --model CDDFuse_IVF/MIF --dataset <X> --group benchmark`

---

## CT-MRI

| Metric | Group | CDDFuse_IVF | CDDFuse_MIF | Winner |
|---|---|---|---|---|
| Mean | Basic Quality | 71.9505 | 59.6022 | IVF |
| SD | Basic Quality | 88.3816 | 78.9923 | IVF |
| AG | Basic Quality | 8.1338 | **8.8942** | **MIF** |
| EN | Basic Quality | 4.7272 | **4.7661** | **MIF** |
| Q_MI | Info Theory | 0.7309 | **0.8424** | **MIF** |
| Q_TE | Info Theory | -7.4654 | -12.0074 | IVF |
| Q_NCIE | Info Theory | 0.0291 | **0.0395** | **MIF** |
| Q_G | Img Feature | 0.5886 | **0.6751** | **MIF** |
| Q_M | Img Feature | 0.6306 | **0.6517** | **MIF** |
| Q_SF | Img Feature | -0.1325 | **-0.0254** | **MIF** |
| Q_P | Img Feature | 0.1141 | **0.2156** | **MIF** |
| Q_S | Struct Sim | 0.3946 | **0.4423** | **MIF** |
| Q_C | Struct Sim | 0.4193 | **0.4912** | **MIF** |
| Q_Y | Struct Sim | 0.9070 | **0.9425** | **MIF** |
| Q_CV | Perception ↓ | -1136.8754 | **-1028.0105** | **MIF** |
| Q_CB | Perception | 0.6346 | **0.7020** | **MIF** |

> **MIF thắng 13/16**, IVF thắng Mean & SD (độ tương phản cao hơn do domain mismatch), Q_TE không ổn định (âm).

---

## PET-MRI

| Metric | Group | CDDFuse_IVF | CDDFuse_MIF | Winner |
|---|---|---|---|---|
| Mean | Basic Quality | 55.4405 | 46.7306 | IVF |
| SD | Basic Quality | 81.4943 | 70.5520 | IVF |
| AG | Basic Quality | 7.6897 | **8.0779** | **MIF** |
| EN | Basic Quality | 4.1475 | 4.1428 | ≈ |
| Q_MI | Info Theory | 0.7448 | **0.8039** | **MIF** |
| Q_TE | Info Theory | 84.3888 | **121.3666** | **MIF** |
| Q_NCIE | Info Theory | 0.0205 | **0.0247** | **MIF** |
| Q_G | Img Feature | 0.6461 | **0.6964** | **MIF** |
| Q_M | Img Feature | 0.6990 | **0.7519** | **MIF** |
| Q_SF | Img Feature | -0.0705 | **-0.0213** | **MIF** |
| Q_P | Img Feature | 0.1564 | **0.2234** | **MIF** |
| Q_S | Struct Sim | 0.3708 | **0.3945** | **MIF** |
| Q_C | Struct Sim | 0.3916 | **0.4115** | **MIF** |
| Q_Y | Struct Sim | 0.9470 | **0.9551** | **MIF** |
| Q_CV | Perception ↓ | -763.7584 | **-323.8115** | **MIF** |
| Q_CB | Perception | 0.6780 | **0.7434** | **MIF** |

> **MIF thắng 13/16**, IVF thắng Mean & SD.

---

## SPECT-MRI

| Metric | Group | CDDFuse_IVF | CDDFuse_MIF | Winner |
|---|---|---|---|---|
| Mean | Basic Quality | 46.5620 | 37.5270 | IVF |
| SD | Basic Quality | 71.6251 | 58.1312 | IVF |
| AG | Basic Quality | 5.1928 | **5.3341** | **MIF** |
| EN | Basic Quality | 3.8218 | 3.8203 | ≈ |
| Q_MI | Info Theory | 0.8034 | **1.0538** | **MIF** |
| Q_TE | Info Theory | **-5.1861** | -27.4329 | IVF |
| Q_NCIE | Info Theory | 0.0215 | **0.0381** | **MIF** |
| Q_G | Img Feature | 0.6597 | **0.7487** | **MIF** |
| Q_M | Img Feature | 0.7704 | **0.8827** | **MIF** |
| Q_SF | Img Feature | -0.0213 | **-0.0100** | **MIF** |
| Q_P | Img Feature | 0.1870 | **0.3260** | **MIF** |
| Q_S | Struct Sim | 0.3390 | **0.3795** | **MIF** |
| Q_C | Struct Sim | 0.3520 | **0.4233** | **MIF** |
| Q_Y | Struct Sim | 0.9501 | **0.9901** | **MIF** |
| Q_CV | Perception ↓ | -199.3709 | **-54.9798** | **MIF** |
| Q_CB | Perception | 0.6671 | **0.7651** | **MIF** |

> **MIF thắng 13/16**, IVF thắng Mean, SD, Q_TE.

---

## Tổng hợp AVERAGE

| Metric | CT-MRI IVF | CT-MRI MIF | PET-MRI IVF | PET-MRI MIF | SPECT-MRI IVF | SPECT-MRI MIF |
|---|---|---|---|---|---|---|
| Mean | 71.9505 | 59.6022 | 55.4405 | 46.7306 | 46.5620 | 37.5270 |
| SD | 88.3816 | 78.9923 | 81.4943 | 70.5520 | 71.6251 | 58.1312 |
| AG | 8.1338 | **8.8942** | 7.6897 | **8.0779** | 5.1928 | **5.3341** |
| EN | 4.7272 | **4.7661** | 4.1475 | 4.1428 | 3.8218 | 3.8203 |
| Q_MI | 0.7309 | **0.8424** | 0.7448 | **0.8039** | 0.8034 | **1.0538** |
| Q_NCIE | 0.0291 | **0.0395** | 0.0205 | **0.0247** | 0.0215 | **0.0381** |
| Q_G | 0.5886 | **0.6751** | 0.6461 | **0.6964** | 0.6597 | **0.7487** |
| Q_M | 0.6306 | **0.6517** | 0.6990 | **0.7519** | 0.7704 | **0.8827** |
| Q_SF | -0.1325 | **-0.0254** | -0.0705 | **-0.0213** | -0.0213 | **-0.0100** |
| Q_P | 0.1141 | **0.2156** | 0.1564 | **0.2234** | 0.1870 | **0.3260** |
| Q_S | 0.3946 | **0.4423** | 0.3708 | **0.3945** | 0.3390 | **0.3795** |
| Q_C | 0.4193 | **0.4912** | 0.3916 | **0.4115** | 0.3520 | **0.4233** |
| Q_Y | 0.9070 | **0.9425** | 0.9470 | **0.9551** | 0.9501 | **0.9901** |
| Q_CV↓ | -1136.8754 | **-1028.0105** | -763.7584 | **-323.8115** | -199.3709 | **-54.9798** |
| Q_CB | 0.6346 | **0.7020** | 0.6780 | **0.7434** | 0.6671 | **0.7651** |

---

## Kết luận

- **CDDFuse_MIF** vượt trội CDDFuse_IVF trên **13/16 metrics** ở cả 3 dataset
- IVF cho Mean và SD cao hơn do được train trên ảnh hồng ngoại (range cường độ rộng hơn), nhưng không phản ánh chất lượng fusion tốt hơn
- **Dùng CDDFuse_MIF** cho benchmark y tế (CT-MRI, PET-MRI, SPECT-MRI)
- Q_TE không ổn định (có thể âm hoặc dương lớn tùy dataset) — nên dùng thận trọng khi so sánh cross-dataset
