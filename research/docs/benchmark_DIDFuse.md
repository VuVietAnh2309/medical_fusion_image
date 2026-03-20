# DIDFuse Benchmark — 16 Metrics

**Model:** DIDFuse (IJCAI 2020) — Deep Image Decomposition for IR-Visible Fusion
**Checkpoint:** `Models/Encoder_weight_IJCAI.pkl`, `Models/Decoder_weight_IJCAI.pkl`
**Fusion mode:** Sum (base encoder features summed)
**Datasets:** CT-MRI (24 imgs) · PET-MRI (24 imgs) · SPECT-MRI (24 imgs)
**Metrics:** 16 (Basic Quality × 4 + Liu 2012 × 12)
**Color space:** Grayscale (Y channel)
**Note:** Trained on IR-Visible, not medical images specifically

---

## CT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 46.8732 |
| SD | Basic Quality | 65.4367 |
| AG | Basic Quality | 6.9745 |
| EN | Basic Quality | 4.6657 |
| Q_MI | Info Theory | 0.7446 |
| Q_TE | Info Theory | 46.0694 |
| Q_NCIE | Info Theory | 0.0241 |
| Q_G | Img Feature | 0.4273 |
| Q_M | Img Feature | 0.6241 |
| Q_SF | Img Feature | -0.2894 |
| Q_P | Img Feature | 0.1100 |
| Q_S | Struct Sim | 0.2965 |
| Q_C | Struct Sim | 0.3279 |
| Q_Y | Struct Sim | 0.7734 |
| Q_CV | Perception | -1915.7973 |
| Q_CB | Perception | 0.3446 |

---

## PET-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 55.2415 |
| SD | Basic Quality | 79.4536 |
| AG | Basic Quality | 7.3722 |
| EN | Basic Quality | 4.8563 |
| Q_MI | Info Theory | 0.6632 |
| Q_TE | Info Theory | 10.7637 |
| Q_NCIE | Info Theory | 0.0230 |
| Q_G | Img Feature | 0.3407 |
| Q_M | Img Feature | 0.5175 |
| Q_SF | Img Feature | -0.2468 |
| Q_P | Img Feature | 0.1415 |
| Q_S | Struct Sim | 0.2912 |
| Q_C | Struct Sim | 0.2988 |
| Q_Y | Struct Sim | 0.6966 |
| Q_CV | Perception | -1358.7512 |
| Q_CB | Perception | 0.3588 |

---

## SPECT-MRI (24 images)

| Metric | Group | Score |
|---|---|---|
| Mean | Basic Quality | 26.9212 |
| SD | Basic Quality | 45.1462 |
| AG | Basic Quality | 4.2384 |
| EN | Basic Quality | 4.2453 |
| Q_MI | Info Theory | 0.7076 |
| Q_TE | Info Theory | 22.7250 |
| Q_NCIE | Info Theory | 0.0211 |
| Q_G | Img Feature | 0.3062 |
| Q_M | Img Feature | 0.5975 |
| Q_SF | Img Feature | -0.2802 |
| Q_P | Img Feature | 0.1253 |
| Q_S | Struct Sim | 0.2333 |
| Q_C | Struct Sim | 0.2436 |
| Q_Y | Struct Sim | 0.7285 |
| Q_CV | Perception | -681.4363 |
| Q_CB | Perception | 0.3080 |

---

## Quan sát

- **Q_MI trung bình khá** (0.74/0.66/0.71) — DIDFuse giữ mutual information ổn mặc dù không train medical
- **SPECT-MRI Mean = 26.92, SD = 45.15** — thấp nhất, ảnh tối hơn source
- **Q_Y trung bình** (0.77/0.70/0.73) — structural similarity thấp hơn EMMA và LRFNet
- **Q_CB thấp** (~0.35) — không bảo toàn màu tốt, phù hợp với model grayscale-oriented
