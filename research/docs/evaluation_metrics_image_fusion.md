# Các thang đo đánh giá trong bài toán Image Fusion

## Tổng quan

Image Fusion **không có ground truth** → metrics đo **mức bảo toàn thông tin** từ ảnh nguồn A, B sang ảnh fusion F. Tất cả **cao hơn = tốt hơn (↑)**.

| Metric | Đo cái gì | Phạm vi |
|---|---|---|
| **VIF** | Thông tin thị giác (pixel-level, mô hình HVS) | [0, 1+] |
| **SCD** | Cân bằng nội dung giữa 2 nguồn | [-2, 2] |
| **Q^(AB/F)** | Bảo toàn cạnh (edge strength + orientation) | [0, 1] |
| **MS-SSIM** | Tương đồng cấu trúc đa tỉ lệ | [0, 1] |
| **FMI** | Bảo toàn đặc trưng (feature-level) | [0, 1+] |

--- 

## 1. VIF — Visual Information Fidelity

> **Câu hỏi:** "Bao nhiêu thông tin thị giác từ ảnh nguồn được giữ lại trong ảnh fusion, theo cách não người nhận thức?"

**Ý tưởng:** Mô hình hóa quá trình nhìn ảnh như bài toán truyền tin. Ảnh nguồn đi qua "kênh méo" (fusion) rồi đến não. VIF = tỉ lệ thông tin não nhận được từ ảnh fusion so với ảnh gốc.

**Công thức:**

```
         Σⱼ  I(C̄ⱼ ; F̄ⱼ | Sⱼ)       ← Thông tin não nhận từ ẢNH FUSION
VIF = ───────────────────────────
         Σⱼ  I(C̄ⱼ ; Ēⱼ | Sⱼ)       ← Thông tin não nhận từ ẢNH GỐC
```

Khai triển dưới mô hình Gaussian:

```
         Σⱼ Σₖ log₂(1 + gⱼ²·sⱼₖ²·Cᵤ / (σᵥ² + σₙ²))
VIF = ─────────────────────────────────────────────────
         Σⱼ Σₖ log₂(1 + sⱼₖ²·Cᵤ / σₙ²)
```

- `j`: sub-band wavelet, `k`: vị trí không gian
- `gⱼ, σᵥ²`: gain và nhiễu của kênh méo (ước lượng từ ảnh fusion)
- `sⱼₖ`: năng lượng cục bộ, `σₙ²`: nhiễu nội tại của não
- `Cᵤ`: covariance matrix của hệ số wavelet

**Diễn giải:** VIF = 1 → giữ 100% thông tin. VIF > 1 → fusion tăng cường thông tin. VIF < 1 → mất thông tin.

Phiên bản cho fusion (2 nguồn): `VIFF = Σⱼ [wᴬⱼ · VIF(A,F,j) + wᴮⱼ · VIF(B,F,j)]`

---

## 2. SCD — Sum of the Correlations of Differences

> **Câu hỏi:** "Khi trừ fusion cho 1 nguồn, phần còn lại có đúng là thông tin của nguồn kia không?"

**Ý tưởng:** Nếu `F = A + B` thì `F - B ≈ A` và `F - A ≈ B`. Đo bằng tương quan Pearson.

**Công thức:**

```
SCD = r(F - B, A) + r(F - A, B)
```

Với hệ số tương quan Pearson:

```
            Σᵢ (Xᵢ - X̄)(Yᵢ - Ȳ)
r(X, Y) = ──────────────────────────
           √[Σᵢ(Xᵢ-X̄)²] · √[Σᵢ(Yᵢ-Ȳ)²]
```

**Diễn giải:**
- SCD = 2 → hoàn hảo (cả 2 thành phần = 1)
- SCD cao → fusion cân bằng, chứa đủ thông tin cả A và B
- SCD thấp → thiên lệch về 1 modality hoặc thêm artifacts

**Ví dụ CT-MRI:**
```
F(tốt):  F - MRI ≈ xương ≈ CT → r ≈ 0.8 | F - CT ≈ mô mềm ≈ MRI → r ≈ 0.8 | SCD ≈ 1.6 ✓
F(lệch): F - MRI ≈ ít info    → r ≈ 0.2 | F - CT ≈ MRI           → r ≈ 0.7 | SCD ≈ 0.9 ✗
```

---

## 3. Q^(AB/F) — Gradient-Based Quality Metric

> **Câu hỏi:** "Bao nhiêu thông tin cạnh (edges) từ A và B được giữ trong F?"

**Ý tưởng:** Tính Sobel gradient cho cả 3 ảnh (A, B, F). So sánh cường độ cạnh (strength) và hướng cạnh (orientation) giữa nguồn và fusion.

**Quy trình:**

```
Bước 1: Sobel gradient → g_X(i,j) = √(sₓ² + sᵧ²),  α_X(i,j) = arctan(sᵧ/sₓ)

Bước 2: Edge strength preservation → G_AF = min(g_A, g_F) / max(g_A, g_F)  ∈ [0,1]
         Edge orientation preservation → Ā_AF = 1 - |α_A - α_F| / (π/2)   ∈ [0,1]

Bước 3: Q_AF(i,j) = Γ_g · G_AF^(k_g) · Γ_α · Ā_AF^(k_α)    ← edge transfer từ A→F
         Q_BF(i,j) tương tự cho B→F
```

**Công thức cuối:**

```
           Σᵢⱼ [Q_AF · w_A + Q_BF · w_B]
Q^(AB/F) = ─────────────────────────────────
                  Σᵢⱼ [w_A + w_B]
```

- Trọng số `w_A = g_A^L`: vùng nhiều cạnh → quan trọng hơn vùng phẳng

**Diễn giải:** Q = 1 → mọi cạnh từ 2 nguồn đều giữ nguyên. Q > 0.5 → tốt. Q < 0.3 → edges bị mờ nhiều.

---

## 4. MS-SSIM — Multi-Scale Structural Similarity

> **Câu hỏi:** "Ảnh fusion có giống ảnh nguồn về mặt cấu trúc không, ở nhiều mức phân giải?"

**Ý tưởng:** Mắt người nhìn ảnh ở nhiều khoảng cách → nhận thức ở nhiều scale. SSIM đánh giá 3 thành phần (luminance, contrast, structure), MS-SSIM mở rộng qua 5 scales.

**SSIM tại 1 scale** (so sánh ảnh x, y trong cửa sổ cục bộ):

```
              (2μₓμᵧ + C₁)(2σₓᵧ + C₂)
SSIM(x,y) = ─────────────────────────────        C₁ = (0.01·255)², C₂ = (0.03·255)²
              (μₓ²+μᵧ²+C₁)(σₓ²+σᵧ²+C₂)
```

- `(2μₓμᵧ+C₁)/(μₓ²+μᵧ²+C₁)` = luminance (sáng giống nhau?)
- `(2σₓσᵧ+C₂)/(σₓ²+σᵧ²+C₂)` = contrast (tương phản giống nhau?)
- `(σₓᵧ+C₃)/(σₓσᵧ+C₃)` = structure (pattern giống nhau?)

**MS-SSIM** — downsample ảnh qua 5 scales, tính contrast + structure ở mỗi scale, luminance ở scale thô nhất:

```
MS-SSIM(x,y) = l_M^(α_M) · Π_{j=1}^{M} c_j^(β_j) · s_j^(γ_j)
```

Trọng số (M=5): Scale 2-3 có trọng số lớn nhất (β₂=β₃≈0.29) → mắt người nhạy nhất ở chi tiết trung bình.

Cho fusion: `MS-SSIM_fusion = [MS-SSIM(A,F) + MS-SSIM(B,F)] / 2`

**Diễn giải:** MS-SSIM → 1 = cấu trúc được bảo toàn hoàn hảo ở mọi scale.

---

## 5. FMI — Feature Mutual Information

> **Câu hỏi:** "Bao nhiêu thông tin đặc trưng (features) từ nguồn được giữ trong fusion?"

**Ý tưởng:** Trích xuất features (gradient/edge/DCT/wavelet) từ cả 3 ảnh, rồi đo mutual information (MI) giữa features nguồn và features fusion.

**Công thức MI:**

```
MI(X, Y) = H(X) + H(Y) - H(X,Y)

H(X) = -Σₓ p(x)·log₂ p(x)              ← Shannon entropy
H(X,Y) = -Σₓ Σᵧ p(x,y)·log₂ p(x,y)    ← Joint entropy
```

**FMI tổng hợp:**

```
FMI = (1/N) · Σ_blocks [MI(f_A, f_F) + MI(f_B, f_F)]
```

- Trích features: `f_X = FeatureExtract(X)` (VD: Sobel gradient)
- Chia thành blocks w×w, tính MI cục bộ, rồi lấy trung bình

**Diễn giải:** FMI cao → features (texture, edges, patterns) từ nguồn được bảo toàn tốt. MI = 0 → không chia sẻ thông tin → fusion mất hết features.

---

## 6. Tổng hợp: Tại sao cần 5 metrics?

Mỗi metric nhạy với **loại lỗi khác nhau**:

```
Lỗi Fusion              │ VIF  │ SCD  │ Q^AB/F │ MS-SSIM │ FMI
─────────────────────────┼──────┼──────┼────────┼─────────┼──────
Ảnh bị mờ (blur)        │  ↓↓  │  ↓   │  ↓↓↓  │   ↓↓   │  ↓↓
Thiên lệch 1 modality   │  ↓   │  ↓↓↓ │   ↓   │    ↓   │  ↓↓
Mất cạnh/edges           │  ↓   │  ↓   │  ↓↓↓  │   ↓↓   │  ↓↓
Méo cấu trúc tổng thể   │  ↓   │  ↓   │   ↓   │  ↓↓↓   │  ↓
Mất texture details      │  ↓↓  │  ↓   │  ↓↓   │   ↓↓   │  ↓↓↓
```

→ Chỉ dùng 1 metric = đánh giá thiên lệch. Cần 5 metrics → đánh giá **toàn diện**.

**FusionMamba dẫn đầu:**

| Task | Metrics dẫn đầu | Ý nghĩa |
|---|---|---|
| **CT-MRI** | SCD, Q^(AB/F), MS-SSIM | Cân bằng, giữ cạnh, cấu trúc chính xác |
| **PET-MRI** | VIF, Q^(AB/F), MS-SSIM | Thông tin thị giác phong phú, edges sắc |
| **SPECT-MRI** | VIF, SCD, Q^(AB/F), FMI | Dẫn **4/5 metrics** — toàn diện nhất |

---

## Tài liệu tham khảo

1. Sheikh & Bovik (2006). "Image information and visual quality." *IEEE TIP*, 15(2).
2. Aslantas & Bendes (2015). "A new image quality metric for image fusion: SCD." *AEU*, 69(12).
3. Xydeas & Petrovic (2000). "Objective image fusion performance measure." *Electronics Letters*, 36(4).
4. Wang et al. (2003). "Multi-scale structural similarity for image quality assessment." *IEEE Asilomar*.
5. Haghighat et al. (2011). "A non-reference image fusion metric based on MI of image features." *C&EE*, 37(5).
