# 18 Thang đo đánh giá chất lượng ảnh Fusion — Trình bày chi tiết

## Mở đầu

Trong bài toán Image Fusion, chúng ta kết hợp hai ảnh nguồn **A** (ví dụ CT, PET, SPECT) và **B** (MRI) thành một ảnh fusion **F** sao cho F giữ lại thông tin bổ sung từ cả hai modality.

**Điểm quan trọng:** Không tồn tại ảnh ground truth (ảnh chuẩn) cho bài toán fusion. Vì vậy, tất cả các thang đo đều là **no-reference** — chúng đánh giá mức độ F bảo toàn thông tin từ A và B, chứ không so với một ảnh "đúng" nào cả.

Chúng tôi sử dụng **18 thang đo** chia thành **7 nhóm**:
- 4 thang đo đầu tiên theo paper FusionMamba (Xie et al., 2024)
- 12 thang đo theo phân loại của Liu et al. (2012) — IEEE TPAMI
- 2 thang đo bổ sung: EN (Shannon Entropy) và PSNR

**Quy ước chung:** Tất cả 18 thang đo đều **cao hơn = tốt hơn** (↑).

### Ký hiệu

- A(i,j), B(i,j), F(i,j): giá trị pixel tại vị trí (i,j), nằm trong [0, 255]
- M × N: kích thước ảnh
- H(X) = −Σ pₖ log₂ pₖ: entropy Shannon (đo lượng thông tin)
- I(X;Y) = H(X) + H(Y) − H(X,Y): mutual information (thông tin tương hỗ)
- μ_X, σ_X, σ_XY: trung bình, độ lệch chuẩn, hiệp phương sai cục bộ trong cửa sổ trượt W×W

---

## Nhóm 1: Structural Similarity — Đo tương đồng cấu trúc

Nhóm này trả lời câu hỏi: **"Ảnh F có giữ được cấu trúc (texture, edge, pattern) của ảnh nguồn không?"**

### 1.1. MS-SSIM — Multi-Scale Structural Similarity

**Nguồn:** Wang et al. (2003)

**Ý tưởng:** Mắt người nhìn ảnh ở nhiều khoảng cách khác nhau, tức là nhận thức ảnh ở nhiều scale. SSIM gốc chỉ đánh giá ở 1 scale, còn MS-SSIM mở rộng qua 5 scale bằng cách lọc thông thấp và downsample ảnh lặp đi lặp lại.

**Công thức:**

```
MS-SSIM(X, Y) = l_M^(α_M) × ∏_{j=1}^{M} c_j^(β_j) × s_j^(γ_j)
```

Trong đó:
- l_M: so sánh luminance (độ sáng) ở scale thô nhất
- c_j: so sánh contrast (độ tương phản) ở scale j
- s_j: so sánh structure (cấu trúc) ở scale j
- M = 5 scale, với trọng số: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
- Scale 2–3 có trọng số lớn nhất → mắt người nhạy nhất ở chi tiết trung bình

**Cho fusion:** MS-SSIM = ½ × [MS-SSIM(A,F) + MS-SSIM(B,F)]

**Khoảng giá trị:** [0, 1]
- 1 = cấu trúc được bảo toàn hoàn hảo ở mọi scale
- Giá trị thực tế trong benchmark: 0.57 – 0.76

---

### 1.2. FMI — Feature Mutual Information

**Nguồn:** Haghighat et al. (2011)

**Ý tưởng:** Thay vì so pixel thô, FMI trích xuất feature (đặc trưng) bằng toán tử Sobel gradient, rồi đo mutual information giữa feature của ảnh nguồn và ảnh fusion. Chia ảnh thành các block w×w, tính MI cục bộ rồi lấy trung bình.

**Công thức:**

```
FMI = (1/N_b) × Σ_{b=1}^{N_b} [I(f_A^b; f_F^b) + I(f_B^b; f_F^b)]
```

Trong đó:
- f_X^b: vector đặc trưng (gradient) của ảnh X trong block b
- N_b: tổng số block
- I(·;·): mutual information

**Khoảng giá trị:** [0, +∞), giá trị thực tế: 0.18 – 0.47
- Cao → feature (texture, edge, pattern) từ nguồn được bảo toàn tốt
- Thấp → fusion mất feature quan trọng

---

## Nhóm 2: Visual Fidelity — Đo độ trung thực thị giác

Nhóm này trả lời câu hỏi: **"Thông tin thị giác từ ảnh nguồn có được giữ trung thực trong F không?"**

### 2.1. VIF — Visual Information Fidelity

**Nguồn:** Sheikh & Bovik (2006)

**Ý tưởng:** Mô hình hóa quá trình nhìn ảnh như bài toán truyền tin. Ảnh nguồn đi qua "kênh méo" (quá trình fusion), rồi đến não người. VIF đo tỉ lệ thông tin não nhận được từ ảnh fusion so với ảnh gốc, sử dụng mô hình Gaussian Scale Mixture trong miền wavelet.

**Công thức:**

```
VIF(X, F) = Σ_j Σ_k log₂(1 + g_j² s_jk² C_u / (σ_v² + σ_n²))
            ─────────────────────────────────────────────────────
            Σ_j Σ_k log₂(1 + s_jk² C_u / σ_n²)
```

Trong đó:
- j: sub-band wavelet, k: vị trí không gian
- g_j, σ_v²: gain và nhiễu của kênh méo (ước lượng từ ảnh fusion)
- s_jk: năng lượng tín hiệu cục bộ
- σ_n²: nhiễu nội tại của hệ thống thị giác người (HVS)
- C_u: covariance matrix của hệ số wavelet

**Cho fusion:** VIF = ½ × [VIF(A,F) + VIF(B,F)]

**Khoảng giá trị:** [0, 1+]
- VIF = 1: giữ 100% thông tin thị giác
- VIF > 1: fusion tăng cường thông tin (hiếm)
- VIF < 1: mất thông tin
- Giá trị thực tế: 0.16 – 0.48

---

### 2.2. SCD — Sum of Correlations of Differences

**Nguồn:** Aslantas & Bendes (2015)

**Ý tưởng:** Nếu F = A + B (fusion lý tưởng), thì F − B ≈ A và F − A ≈ B. SCD kiểm tra điều này bằng hệ số tương quan Pearson.

**Công thức:**

```
SCD = r(F − B, A) + r(F − A, B)
```

Trong đó r(X, Y) là hệ số tương quan Pearson:
```
r(X, Y) = Σ(Xᵢ − X̄)(Yᵢ − Ȳ) / [√Σ(Xᵢ−X̄)² × √Σ(Yᵢ−Ȳ)²]
```

**Khoảng giá trị:** [−2, 2]
- SCD = 2: hoàn hảo (cả hai thành phần đều = 1)
- Giá trị thực tế: 0.19 – 1.87
- SCD cao → F cân bằng, chứa đủ thông tin cả A và B
- SCD thấp → F thiên lệch về 1 modality hoặc thêm artifact

**Ví dụ CT-MRI:**
- F(tốt): F − MRI ≈ xương ≈ CT → r ≈ 0.8; F − CT ≈ mô mềm ≈ MRI → r ≈ 0.8 → SCD ≈ 1.6 ✓
- F(lệch): F thiên về MRI → F − CT ≈ MRI → r cao, nhưng F − MRI ≈ 0 → r thấp → SCD ≈ 0.9 ✗

---

## Nhóm 3: Information Theory — Lý thuyết thông tin

Nhóm này trả lời câu hỏi: **"Bao nhiêu thông tin thống kê từ A và B được truyền sang F?"**

Ba thang đo sau thuộc Group 1 trong phân loại của Liu et al. (2012).

### 3.1. Q_MI — Normalized Mutual Information

**Nguồn:** Hossny et al. (2008)

**Ý tưởng:** Mutual Information (MI) đo lượng thông tin chung giữa hai biến. Q_MI chuẩn hóa MI để tránh phụ thuộc vào scale, cho phép so sánh giữa các dataset khác nhau.

**Công thức:**

```
Q_MI = 2 × [I(A,F)/(H(A)+H(F)) + I(B,F)/(H(B)+H(F))]
```

Trong đó:
- I(X,Y) = H(X) + H(Y) − H(X,Y): mutual information
- H(X) = −Σ p(x) log₂ p(x): entropy Shannon
- H(X,Y) = −Σ p(x,y) log₂ p(x,y): joint entropy (tính từ joint histogram)

**Khoảng giá trị:** [0, 2]
- Q_MI = 2: F chứa toàn bộ thông tin từ cả A và B
- Giá trị thực tế: 0.53 – 1.05

---

### 3.2. Q_TE — Tsallis Entropy-Based Metric

**Nguồn:** Cvejic et al. (2006), Nava et al. (2007)

**Ý tưởng:** Sử dụng entropy Tsallis (q = 1.85) thay cho Shannon. Tsallis entropy nhạy hơn với phân phối hiếm (rare events), phù hợp cho ảnh y tế có nhiều vùng đồng nhất và ít vùng chi tiết.

**Công thức:**

```
Q_TE = [I^q(A,F) + I^q(B,F)] / [H^q(A) + H^q(B) − I^q(A,B)]
```

Trong đó:
- H^q(X) = (1/(q−1)) × (1 − Σ pₖ^q): Tsallis entropy
- I^q(X,Y): Tsallis mutual information
- q = 1.85 (tham số Tsallis)

**Khoảng giá trị:** (−∞, +∞), **không bị giới hạn**
- Mẫu số có thể âm khi A và B tương quan cao → Q_TE âm
- Giá trị dương lớn (10–134): A, B bổ sung cho nhau → tốt
- Giá trị âm (−32 đến −7): A, B tương quan mạnh (thường xảy ra với CT-MRI)
- **Chỉ nên dùng để so sánh tương đối** giữa các model trên cùng dataset

---

### 3.3. Q_NCIE — Nonlinear Correlation Information Entropy

**Nguồn:** Liu et al. (2012)

**Ý tưởng:** Tính hệ số tương quan phi tuyến (NCC) giữa từng cặp trong bộ ba {A, B, F}, tạo ma trận tương quan 3×3, rồi lấy entropy của eigenvalue để đo mức độ F tham gia đều vào cấu trúc tương quan với cả A và B.

**Công thức:**

```
NCC(X,Y) = H'(X) + H'(Y) − H'(X,Y)    (H' dùng log base b=256)

R = | 1        NCC_AB   NCC_AF |
    | NCC_BA   1        NCC_BF |
    | NCC_FA   NCC_FB   1      |

Q_NCIE = 1 + Σ_{i=1}^{3} (λᵢ/3) × log₃(λᵢ/3)
```

Trong đó λ₁, λ₂, λ₃ là eigenvalue của ma trận R.

**Khoảng giá trị:** [0, 1]
- Giá trị thực tế: 0.016 – 0.041
- Cao → F tham gia đồng đều vào tương quan với cả A và B
- Thấp → F thiên về một nguồn hoặc không liên quan

---

## Nhóm 4: Image Feature — Đặc trưng ảnh

Nhóm này trả lời câu hỏi: **"Các cạnh (edges), kết cấu (textures), chi tiết không gian của A và B có được giữ trong F không?"**

Bốn thang đo sau thuộc Group 2 trong Liu et al. (2012).

### 4.1. Q_G (hay Q^{AB/F}) — Gradient-Based Quality

**Nguồn:** Xydeas & Petrovic (2000)

**Ý tưởng:** Đây là thang đo bảo toàn cạnh được sử dụng **phổ biến nhất**. Tính Sobel gradient cho cả 3 ảnh, so sánh cường độ cạnh và hướng cạnh giữa nguồn và fusion. Các pixel có cạnh mạnh được đánh trọng số cao hơn.

**Công thức:**

```
Bước 1: Sobel gradient
  g_X(i,j) = √(s_x² + s_y²)      (cường độ cạnh)
  α_X(i,j) = arctan(s_y / s_x)    (hướng cạnh)

Bước 2: Bảo toàn cường độ và hướng
  G^AF = min(g_A, g_F) / max(g_A, g_F)     ∈ [0,1]
  Δ^AF = 1 − |α_A − α_F| / (π/2)          ∈ [0,1]

Bước 3: Qua hàm sigmoid
  Q_g^AF = T_g / (1 + exp(k_g × (G^AF − D_g)))
  Q_α^AF = T_a / (1 + exp(k_a × (Δ^AF − D_a)))
  Q^AF = Q_g^AF × Q_α^AF

Bước 4: Trung bình có trọng số
  Q_G = Σ[Q^AF × g_A + Q^BF × g_B] / Σ[g_A + g_B]
```

Tham số sigmoid: T_g=0.9994, k_g=−15, D_g=0.5; T_a=0.9879, k_a=−22, D_a=0.8

**Khoảng giá trị:** [0, 1]
- Giá trị thực tế: 0.23 – 0.75
- Q_G > 0.5: tốt; Q_G < 0.3: mất cạnh đáng kể

---

### 4.2. Q_M — Multi-Scale Wavelet Metric

**Nguồn:** Wang & Liu (2008)

**Ý tưởng:** Dùng biến đổi Haar wavelet 2 mức. Ở mỗi scale, so sánh hệ số tần số cao (LH, HL, HH) giữa nguồn và fusion, đánh trọng số theo năng lượng nguồn.

**Công thức:**

```
EP_s^AF = (1/3) × Σ_d mean[exp(−|d_A^s − d_F^s|)]     (d = LH, HL, HH)

Q_s^{AB/F} = Σ[EP_s^AF × w_A^s + EP_s^BF × w_B^s] / Σ[w_A^s + w_B^s]

Q_M = ∏_{s=1}^{S} (Q_s)^{α_s}
```

**Khoảng giá trị:** [0, 1]
- Giá trị thực tế: 0.51 – 0.88
- Đo bảo toàn feature ở cả chi tiết mịn (fine) và thô (coarse)

---

### 4.3. Q_SF — Spatial Frequency Ratio

**Nguồn:** Zheng et al. (2007)

**Ý tưởng:** Spatial Frequency đo mức độ hoạt động (activity) của ảnh thông qua gradient theo 4 hướng. Q_SF so sánh SF của ảnh fusion với SF tham chiếu (tính từ gradient lớn nhất của A và B).

**Công thức:**

```
SF = √(RF² + CF² + MDF² + SDF²)

RF, CF: gradient theo hàng, cột
MDF, SDF: gradient theo đường chéo chính, phụ

SF_ref: tính từ max gradient của A và B theo 4 hướng

Q_SF = (SF(F) − SF_ref) / SF_ref
```

**Khoảng giá trị:** (−1, +∞)
- Q_SF = 0: F có mức chi tiết bằng nguồn tốt nhất → lý tưởng
- Q_SF < 0: F mượt hơn nguồn → over-smoothing (phổ biến)
- Q_SF > 0: F có nhiều chi tiết hơn nguồn → enhancement
- Giá trị thực tế: −0.48 đến −0.003
- **Giá trị âm là bình thường** trong medical fusion

---

### 4.4. Q_P — Phase Congruency-Based Metric

**Nguồn:** Zhao et al. (2007)

**Ý tưởng:** Trích xuất 3 loại feature map: gradient magnitude (g), Laplacian (ℓ), và orientation (φ). Tính tương quan chéo tối đa giữa feature nguồn và feature fusion. Dùng quy tắc "max": F được thưởng nếu match với **bất kỳ** nguồn nào.

**Công thức:**

```
ρ(F_a, F_b, F_f) = max(corr(F_a, F_f), corr(F_b, F_f), corr(F_a+F_b, F_f))

Q_P = ρ(g_A, g_B, g_F) × ρ(ℓ_A, ℓ_B, ℓ_F) × ρ(φ_A, φ_B, φ_F)
```

**Khoảng giá trị:** [0, 1]
- Giá trị thực tế: 0.08 – 0.34
- Cao → các feature nhận thức (cạnh, góc, đường gờ) được bảo toàn
- Tích 3 thành phần → nghiêm ngặt: cả 3 loại feature đều phải tốt

---

## Nhóm 5: Structural Similarity — Tương đồng cấu trúc (Liu 2012)

Nhóm này trả lời câu hỏi: **"Cấu trúc tổng thể của F có giống A và B không?"**

Ba thang đo sau thuộc Group 3 trong Liu et al. (2012), đều dựa trên UIQI (Universal Image Quality Index):

```
Q₀(X,Y) = (2μ_Xμ_Y)/(μ_X²+μ_Y²) × (2σ_Xσ_Y)/(σ_X²+σ_Y²) × σ_XY/(σ_Xσ_Y)
           \_____________/             \_____________/             \________/
              luminance                   contrast                 structure
```

### 5.1. Q_S — Piella's Saliency-Weighted Metric

**Nguồn:** Piella & Heijmans (2003)

**Ý tưởng:** Tại mỗi vị trí, nguồn nào "hoạt động" hơn (variance cao hơn) thì được đánh trọng số lớn hơn. Ví dụ: tại vùng xương trong CT-MRI, σ²_CT >> σ²_MRI → ưu tiên so sánh CT-F.

**Công thức:**

```
λ(w) = σ²_A(w) / [σ²_A(w) + σ²_B(w)]

Q_S = (1/|W|) × Σ_w [λ(w) × Q₀(A,F|w) + (1−λ(w)) × Q₀(B,F|w)]
```

**Khoảng giá trị:** [−1, 1], thực tế [0, 1]
- Giá trị thực tế: 0.22 – 0.62

---

### 5.2. Q_C — Cvejic's Covariance-Weighted Metric

**Nguồn:** Cvejic et al. (2005)

**Ý tưởng:** Thay trọng số dựa trên variance (Q_S) bằng covariance giữa mỗi nguồn và F. Nguồn nào F tương quan mạnh hơn → được đánh trọng số cao hơn.

**Công thức:**

```
sim(w) = σ_AF(w) / [σ_AF(w) + σ_BF(w)]

Q_C = (1/|W|) × Σ_w [sim(w) × Q₀(A,F|w) + (1−sim(w)) × Q₀(B,F|w)]
```

**Khoảng giá trị:** [−1, 1], thực tế [0, 1]
- Giá trị thực tế: 0.24 – 0.62
- Thưởng cho fusion biết chọn nguồn phù hợp ở từng vùng

---

### 5.3. Q_Y — Yang's SSIM-Based Adaptive Metric

**Nguồn:** Yang et al. (2008)

**Ý tưởng:** Dùng SSIM với ngưỡng thích ứng. Nếu A và B giống nhau tại vị trí đó → F phải match cả hai. Nếu A và B rất khác nhau (điển hình cho CT-MRI, PET-MRI) → F chỉ cần match nguồn tốt hơn.

**Công thức:**

```
Q_Y(w) = {
  λ(w) × SSIM(A,F|w) + (1−λ(w)) × SSIM(B,F|w),    nếu SSIM(A,B|w) ≥ 0.75
  max(SSIM(A,F|w), SSIM(B,F|w)),                     nếu SSIM(A,B|w) < 0.75
}
```

**Khoảng giá trị:** [−1, 1], thực tế [0, 1]
- Giá trị thực tế: 0.52 – 0.99
- Ngưỡng 0.75 rất phù hợp cho multimodal medical fusion vì A và B thường rất khác nhau

---

## Nhóm 6: Human Perception — Nhận thức con người

Nhóm này trả lời câu hỏi: **"Ảnh F có nhìn đúng với mắt người không?"**

Hai thang đo sau thuộc Group 4 trong Liu et al. (2012), mô hình hóa hệ thống thị giác người (HVS).

### 6.1. Q_CV — Chen-Varshney Perceptual Distortion Metric

**Nguồn:** Chen & Varshney (2007)

**Ý tưởng:** 5 bước:
1. Trích cạnh bằng Sobel → edge strength map G_K
2. Chia ảnh thành các block không chồng lấp
3. Tính saliency (độ nổi bật) của mỗi block: Λ_X(b) = Σ g_X^α (α=5) → vùng nhiều cạnh = quan trọng hơn
4. Lọc CSF (Mannos-Sakrison) → loại bỏ thành phần mắt không nhìn thấy
5. Tính distortion D_X(b) từ ảnh đã lọc CSF

**Công thức:**

```
Q_CV = − Σ_b [Λ_A(b)×D_A(b) + Λ_B(b)×D_B(b)] / Σ_b [Λ_A(b) + Λ_B(b)]
```

Dấu âm được thêm để quy ước "cao hơn = tốt hơn".

**Khoảng giá trị:** (−∞, 0]
- Q_CV = 0: không có distortion → hoàn hảo
- Giá trị thực tế: −3927 đến −55
- **Tại sao giá trị âm rất lớn?** Ví dụ LRFNet (Q_CV = −3927) hoặc NestFuse (Q_CV = −3497): ảnh fusion có phân bố cường độ rất khác so với ảnh nguồn tại các vùng có nhiều cạnh (salient regions) → (X−F)² lớn → distortion lớn → Q_CV âm lớn.
- CDDFuse_MIF (Q_CV = −55): distortion thấp → ảnh fusion gần với nguồn hơn.

---

### 6.2. Q_CB — Chen-Blum Perceptual Contrast Metric

**Nguồn:** Chen & Blum (2009)

**Ý tưởng:** Mô hình cảm nhận contrast (độ tương phản) theo kiểu band-pass của mắt, sử dụng bộ lọc CSF dạng Difference-of-Gaussians (DoG). 5 bước:
1. Lọc CSF trong miền tần số (DoG filter)
2. Tính local contrast: C(i,j) = |φ_k ∗ I| / |φ_{k+1} ∗ I| − 1
3. Masking: C'_A = t × C_A^p / (h × C_A^q + Z) → mô hình contrast masking
4. Tính saliency: λ_A = C'_A² / (C'_A² + C'_B²)
5. So sánh contrast giữa nguồn và fusion: Q_CM

**Công thức:**

```
Q_CM(X, F) = (2C_X×C_F + Z) / (C_X² + C_F² + Z)

Q_CB = Σ[w_A × Q_CM(A,F) + w_B × Q_CM(B,F)] / Σ[w_A + w_B]
```

**Khoảng giá trị:** [0, 1]
- Giá trị thực tế: 0.31 – 0.77
- Q_CB > 0.6: tốt
- Q_CB < 0.4: mất contrast đáng kể

---

## Nhóm 7: Supplementary — Thang đo bổ sung

Hai thang đo sau được sử dụng rộng rãi trong các benchmark và được thêm vào theo yêu cầu đánh giá bổ sung.

### 7.1. EN — Shannon Entropy

**Nguồn:** Shannon (1948), được sử dụng phổ biến trong VIFB benchmark và nhiều paper fusion.

**Ý tưởng:** Đo lượng thông tin (information richness) chứa trong ảnh fusion F. Entropy cao nghĩa là pixel trải đều trên nhiều mức xám → ảnh chứa nhiều thông tin. Entropy thấp nghĩa là ảnh đồng nhất (ví dụ: ảnh trắng hoặc đen).

**Công thức:**

```
EN = −Σ_{k=0}^{255} p_k × log₂(p_k)
```

Trong đó:
- p_k: xác suất xuất hiện của mức xám k (tính từ histogram của F)
- Tổng trên 256 bins (0–255)

**Khoảng giá trị:** [0, 8] bits
- EN = 0: ảnh hoàn toàn đồng nhất (1 mức xám)
- EN = 8: phân bố đều hoàn hảo (uniform distribution)
- Giá trị thực tế: 3.68 – 6.39

**Lưu ý quan trọng:** EN là **single-image metric** — chỉ đo ảnh F, không so sánh với ảnh nguồn A, B. Một ảnh noise ngẫu nhiên có EN rất cao nhưng fusion tệ. Vì vậy EN chỉ mang tính tham khảo, không nên dùng làm tiêu chí chính.

---

### 7.2. PSNR — Peak Signal-to-Noise Ratio

**Nguồn:** Metric kinh điển trong image processing. Wang & Bovik (2004, 2009) đã chứng minh hạn chế của PSNR, nhưng vẫn được sử dụng rộng rãi.

**Ý tưởng:** Đo tỉ lệ giữa giá trị pixel tối đa và sai số trung bình bình phương (MSE) giữa ảnh gốc và ảnh so sánh. PSNR cao → sai số pixel thấp → hai ảnh giống nhau.

**Công thức:**

```
MSE(X, F) = (1/MN) × Σ_{i,j} [X(i,j) − F(i,j)]²

PSNR(X, F) = 10 × log₁₀(L² / MSE)     (L = 255 cho ảnh 8-bit)
```

**Cho fusion:** PSNR = ½ × [PSNR(A,F) + PSNR(B,F)]

**Khoảng giá trị:** [0, +∞) dB
- Giá trị thực tế trong benchmark: 13.5 – 25.7 dB

**Hạn chế trong bài toán Image Fusion:**

PSNR là **full-reference metric** — yêu cầu ảnh tham chiếu (ground truth). Trong image fusion không có ground truth, nên phải tính trung bình PSNR(A,F) và PSNR(B,F). Điều này dẫn đến **3 vấn đề**:

1. **Phạt fusion tốt:** Nếu model fusion tốt, F chứa thông tin từ cả A và B → F khác cả A lẫn B → MSE cao → PSNR thấp. Nghịch lý: fusion càng tốt, PSNR càng thấp.

2. **Không tương quan với nhận thức người:** Wang & Bovik (2004) Figure 2 chứng minh nhiều ảnh có cùng MSE/PSNR nhưng chất lượng nhận thức hoàn toàn khác nhau. Wang & Bovik (2009) Section "SO WHAT'S WRONG WITH THE MSE?" liệt kê 4 giả định sai của MSE.

3. **Các paper hàng đầu không dùng PSNR cho fusion:** FusionMamba (2024), CDDFuse (CVPR 2023), DDFM (ICCV 2023), Liu et al. (2012) — không paper nào dùng PSNR làm metric chính.

**Kết luận:** PSNR được đưa vào benchmark để tham khảo và đáp ứng yêu cầu đánh giá đầy đủ, nhưng **không nên dùng làm tiêu chí chính** trong bài toán image fusion. PSNR phù hợp hơn cho super-resolution, denoising, compression (có reference image).

**Tài liệu tham khảo:**
- Wang, Bovik, Sheikh & Simoncelli (2004) — *"Image Quality Assessment: From Error Visibility to Structural Similarity"*, IEEE TIP — Figure 2, trang 4
- Wang & Bovik (2009) — *"Mean Squared Error: Love It or Leave It?"*, IEEE Signal Processing Magazine — Figure 2 trang 100, Section "SO WHAT'S WRONG WITH THE MSE?" trang 100

---

## Tổng hợp 18 thang đo

| # | Thang đo | Nhóm | Khoảng | Đo cái gì |
|---|----------|------|--------|-----------|
| 1 | MS-SSIM | Similarity | [0, 1] | Tương đồng cấu trúc đa tỉ lệ |
| 2 | FMI | Similarity | [0, +∞) | Thông tin đặc trưng được bảo toàn |
| 3 | VIF | Fidelity | [0, 1+] | Trung thực thông tin thị giác |
| 4 | SCD | Fidelity | [−2, 2] | Cân bằng nội dung hai nguồn |
| 5 | Q_MI | Info Theory | [0, 2] | Thông tin tương hỗ chuẩn hóa |
| 6 | Q_TE | Info Theory | (−∞, +∞) | MI dựa trên Tsallis entropy |
| 7 | Q_NCIE | Info Theory | [0, 1] | Entropy tương quan phi tuyến |
| 8 | Q^{AB/F} | Image Feature | [0, 1] | Bảo toàn cạnh (gradient) |
| 9 | Q_M | Image Feature | [0, 1] | Bảo toàn feature đa tỉ lệ (wavelet) |
| 10 | Q_SF | Image Feature | (−1, +∞) | Tỉ lệ tần số không gian |
| 11 | Q_P | Image Feature | [0, 1] | Bảo toàn feature nhận thức |
| 12 | Q_S | Struct Sim | [−1, 1] | UIQI đánh trọng số theo saliency |
| 13 | Q_C | Struct Sim | [−1, 1] | UIQI đánh trọng số theo covariance |
| 14 | Q_Y | Struct Sim | [−1, 1] | SSIM thích ứng |
| 15 | Q_CV | Perception | (−∞, 0] | Distortion nhận thức (gần 0 = tốt) |
| 16 | Q_CB | Perception | [0, 1] | Bảo toàn contrast nhận thức |
| 17 | EN | Supplementary | [0, 8] | Entropy (lượng thông tin ảnh F) |
| 18 | PSNR | Supplementary | [0, +∞) dB | Tỉ số tín hiệu/nhiễu đỉnh |

**Mỗi thang đo nhạy với loại lỗi khác nhau:**

| Loại lỗi | Thang đo nhạy nhất |
|----------|-------------------|
| Ảnh bị mờ (blur) | Q^{AB/F}, Q_SF, VIF |
| Thiên lệch 1 modality | SCD, Q_S, Q_C |
| Mất cạnh/edges | Q^{AB/F}, Q_P, FMI |
| Méo cấu trúc tổng thể | MS-SSIM, Q_Y, Q_M |
| Mất texture details | FMI, VIF, Q_P |
| Méo contrast nhận thức | Q_CB, Q_CV |
| Thông tin nghèo nàn | EN |

→ **Không thang đo nào đủ một mình. Cần đánh giá toàn diện trên nhiều thang đo.**
