"""
Image Fusion Metrics — Standardized Implementation
====================================================
Sources:
  [1] Liu et al. (2012) "Objective Assessment of Multiresolution Image Fusion
      Algorithms" IEEE TPAMI, Vol.34, No.1 — 12 metrics in 4 groups
  [2] VIFB: https://github.com/xingchenzhang/VIFB/tree/master/metrics
      — Reference MATLAB implementations for standardization
  [3] FusionMamba paper additional metrics (VIF, SCD, MS-SSIM, FMI)

Metrics implemented (all higher is better unless noted):
  ── Liu 2012 Paper (4 groups, 12 metrics) ──
  Group 1 - Information Theory:    Q_MI, Q_TE, Q_NCIE
  Group 2 - Image Feature:         Q_G (=Qabf), Q_M, Q_SF, Q_P
  Group 3 - Structural Similarity: Q_S, Q_C, Q_Y
  Group 4 - Human Perception:      Q_CV, Q_CB

  ── VIFB Additional ──
  EN, MI, AG, SF, SSIM, CE (↓lower=better), PSNR, EI

  ── FusionMamba Paper Additional ──
  VIF, SCD, MS_SSIM, FMI

Usage:
    python metrics_liu2012.py --dataset PET-MRI
    python metrics_liu2012.py --fused_dir ./outputs/PET-MRI \
        --source1_dir <path> --source2_dir <path>
    python metrics_liu2012.py --dataset PET-MRI --group liu2012
    python metrics_liu2012.py --dataset PET-MRI --group vifb
    python metrics_liu2012.py --dataset PET-MRI --group all
"""

import os
import cv2
import argparse
import warnings
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.signal import fftconvolve

warnings.filterwarnings('ignore')


# ============================================================
#  Utility functions
# ============================================================

def _normalize255(img):
    """Normalize image to [0, 255] range (matching VIFB normalize1.m)."""
    img = img.astype(np.float64)
    mn, mx = img.min(), img.max()
    if mx == mn:
        return img
    return np.round((img - mn) / (mx - mn) * 255)


def _entropy_log2(img):
    """Shannon entropy H(X) using log2, 256 bins (VIFB metricsEntropy.m)."""
    img = np.floor(img.astype(np.float64)).astype(np.int32)
    hist = np.bincount(img.ravel().clip(0, 255), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _mutual_info_ln(a, b):
    """MI using natural log, normalized to [0,255] (VIFB metricsMutinf.m)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    # Normalize to [0,1] then scale to [0,255] int
    if a.max() != a.min():
        a = (a - a.min()) / (a.max() - a.min())
    else:
        a = np.zeros_like(a)
    if b.max() != b.min():
        b = (b - b.min()) / (b.max() - b.min())
    else:
        b = np.zeros_like(b)
    a = np.clip(np.round(a * 255).astype(np.int32), 0, 255)
    b = np.clip(np.round(b * 255).astype(np.int32), 0, 255)

    # Joint & marginal histograms
    hab = np.zeros((256, 256), dtype=np.float64)
    np.add.at(hab, (a.ravel(), b.ravel()), 1)

    ha = hab.sum(axis=1)
    hb = hab.sum(axis=0)
    total = hab.sum()

    # Entropies using natural log
    p_ab = hab / total
    p_a = ha / total
    p_b = hb / total

    Ha = -np.sum(p_a[p_a > 0] * np.log(p_a[p_a > 0]))
    Hb = -np.sum(p_b[p_b > 0] * np.log(p_b[p_b > 0]))
    Hab = -np.sum(p_ab[p_ab > 0] * np.log(p_ab[p_ab > 0]))

    return Ha + Hb - Hab


def _mutual_info_log2(a, b):
    """MI using log2 for information-theoretic metrics."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.max() != a.min():
        a = (a - a.min()) / (a.max() - a.min())
    else:
        a = np.zeros_like(a)
    if b.max() != b.min():
        b = (b - b.min()) / (b.max() - b.min())
    else:
        b = np.zeros_like(b)
    a = np.clip(np.round(a * 255).astype(np.int32), 0, 255)
    b = np.clip(np.round(b * 255).astype(np.int32), 0, 255)

    hab = np.zeros((256, 256), dtype=np.float64)
    np.add.at(hab, (a.ravel(), b.ravel()), 1)
    ha = hab.sum(axis=1)
    hb = hab.sum(axis=0)
    total = hab.sum()

    p_ab = hab / total
    p_a = ha / total
    p_b = hb / total

    Ha = -np.sum(p_a[p_a > 0] * np.log2(p_a[p_a > 0]))
    Hb = -np.sum(p_b[p_b > 0] * np.log2(p_b[p_b > 0]))
    Hab = -np.sum(p_ab[p_ab > 0] * np.log2(p_ab[p_ab > 0]))

    return Ha + Hb - Hab, Ha, Hb, Hab


def _sobel_edge(img):
    """Sobel gradient: edge strength g and orientation alpha."""
    img = img.astype(np.float64)
    # Sobel kernels matching VIFB: h3 for x, h1 for y
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    sx = fftconvolve(img, h3, mode='same')
    sy = fftconvolve(img, h1, mode='same')
    g = np.sqrt(sx ** 2 + sy ** 2)
    # Avoid division by zero (matching VIFB: add 0.00001 where sx==0)
    sx_safe = sx.copy()
    sx_safe[sx_safe == 0] = 1e-5
    alpha = np.arctan(sy / sx_safe)
    return g, alpha


def _gaussian_kernel(size, sigma):
    """2D Gaussian kernel."""
    x = np.arange(-(size // 2), size // 2 + 1)
    y = np.arange(-(size // 2), size // 2 + 1)
    X, Y = np.meshgrid(x, y)
    G = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return G


def _ssim_single(img1, fused, K1=0.01, K2=0.03, L=255, win_size=11, sigma=1.5):
    """SSIM between two images (VIFB metricsSsim.m)."""
    img1 = img1.astype(np.float64)
    fused = fused.astype(np.float64)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    w = _gaussian_kernel(win_size, sigma)
    w = w / w.sum()

    mu_a = fftconvolve(img1, w, mode='valid')
    mu_b = fftconvolve(fused, w, mode='valid')

    sigma_a_sq = fftconvolve(img1 * img1, w, mode='valid') - mu_a ** 2
    sigma_b_sq = fftconvolve(fused * fused, w, mode='valid') - mu_b ** 2
    sigma_ab = fftconvolve(img1 * fused, w, mode='valid') - mu_a * mu_b

    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a_sq + sigma_b_sq + C2))
    return np.mean(ssim_map)


def _uiqi_map(a, b, win_size=8):
    """Universal Image Quality Index (UIQI / Q) map."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = uniform_filter(a, size=win_size)
    mu_b = uniform_filter(b, size=win_size)
    sigma_a_sq = np.maximum(uniform_filter(a * a, size=win_size) - mu_a ** 2, 0)
    sigma_b_sq = np.maximum(uniform_filter(b * b, size=win_size) - mu_b ** 2, 0)
    sigma_ab = uniform_filter(a * b, size=win_size) - mu_a * mu_b
    num = 4 * sigma_ab * mu_a * mu_b
    den = (sigma_a_sq + sigma_b_sq) * (mu_a ** 2 + mu_b ** 2) + 1e-10
    return num / den


def _local_variance(img, win_size=8):
    """Local variance."""
    img = img.astype(np.float64)
    mu = uniform_filter(img, size=win_size)
    return np.maximum(uniform_filter(img ** 2, size=win_size) - mu ** 2, 0)


# ============================================================
# GROUP 1: Information Theory-Based Metrics (Liu 2012)
# ============================================================

def q_mi(A, B, F):
    """Q_MI: Normalized Mutual Information — Hossny's definition (Liu 2012, Eq.4).

    Q_MI = 2 * [ MI(A,F)/(H(A)+H(F)) + MI(B,F)/(H(B)+H(F)) ]
    """
    mi_af, ha, _, haf = _mutual_info_log2(A, F)
    mi_bf, _, hb, hbf = _mutual_info_log2(B, F)
    hf = _entropy_log2(F)
    ha = _entropy_log2(A)
    hb = _entropy_log2(B)

    return 2.0 * (mi_af / (ha + hf + 1e-10) + mi_bf / (hb + hf + 1e-10))


def q_te(A, B, F, q_param=1.85):
    """Q_TE: Tsallis Entropy fusion metric, normalized (Liu 2012, Eq.7).

    Q_TE = [I^q(A,F) + I^q(B,F)] / [H^q(A) + H^q(B) - I^q(A,B)]
    """
    n_bins = 256

    def _tsallis_entropy(X, q_val):
        hist = np.histogram(X.ravel(), bins=n_bins, range=(0, 256))[0].astype(np.float64)
        px = hist / hist.sum()
        px = px[px > 0]
        return (1.0 / (q_val - 1.0)) * (1.0 - np.sum(px ** q_val))

    def _tsallis_mi(X, Y, q_val):
        hist_2d = np.histogram2d(X.ravel().astype(np.float64),
                                  Y.ravel().astype(np.float64),
                                  bins=n_bins, range=[[0, 256], [0, 256]])[0]
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # Vectorized computation (Eq. 5)
        nz = (pxy > 0) & (px[:, None] * py[None, :] > 0)
        marginal = px[:, None] * py[None, :]
        val = np.sum((pxy[nz] ** q_val) / (marginal[nz] ** (q_val - 1)))
        return (1.0 / (q_val - 1.0)) * (val - 1.0)

    iq_af = _tsallis_mi(A, F, q_param)
    iq_bf = _tsallis_mi(B, F, q_param)
    hq_a = _tsallis_entropy(A, q_param)
    hq_b = _tsallis_entropy(B, q_param)
    iq_ab = _tsallis_mi(A, B, q_param)

    den = hq_a + hq_b - iq_ab
    if abs(den) < 1e-10:
        return 0.0
    return (iq_af + iq_bf) / den


def q_ncie(A, B, F):
    """Q_NCIE: Nonlinear Correlation Information Entropy (Liu 2012, Eq.13).

    Builds 3x3 NCC matrix R, eigenvalues → Q_NCIE = 1 + sum(λ_i/3 * log_3(λ_i/3))
    """
    n_bins = 256
    log_b = np.log(n_bins)

    def _entropy_base_b(img):
        hist = np.histogram(img.ravel(), bins=n_bins, range=(0, 256))[0].astype(np.float64)
        p = hist / hist.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p)) / log_b

    def _joint_entropy_base_b(x, y):
        h2d = np.histogram2d(x.ravel().astype(np.float64), y.ravel().astype(np.float64),
                              bins=n_bins, range=[[0, 256], [0, 256]])[0]
        p = h2d / h2d.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p)) / log_b

    def _ncc(x, y):
        return _entropy_base_b(x) + _entropy_base_b(y) - _joint_entropy_base_b(x, y)

    ncc_ab = _ncc(A, B)
    ncc_af = _ncc(A, F)
    ncc_bf = _ncc(B, F)

    R = np.array([
        [1.0,    ncc_ab, ncc_af],
        [ncc_ab, 1.0,    ncc_bf],
        [ncc_af, ncc_bf, 1.0]
    ])

    eigenvalues = np.maximum(np.linalg.eigvalsh(R), 1e-10)
    p = eigenvalues / 3.0
    return 1.0 + np.sum(p * np.log(p) / np.log(3))


# ============================================================
# GROUP 2: Image Feature-Based Metrics (Liu 2012)
# ============================================================

def q_g(A, B, F):
    """Q_G / Qabf: Gradient-based quality (Xydeas & Petrovic, Liu 2012 Eq.21).

    Standardized with VIFB metricsQabf.m — same parameters and formula.
    """
    pA = _normalize255(A)
    pB = _normalize255(B)
    pF = _normalize255(F)

    gA, aA = _sobel_edge(pA)
    gB, aB = _sobel_edge(pB)
    gF, aF = _sobel_edge(pF)

    # Parameters (matching VIFB exactly)
    L, Tg, kg, Dg = 1, 0.9994, -15, 0.5
    Ta, ka, Da = 0.9879, -22, 0.8

    def _q_pair(gs, als, gf_, alf_):
        GAF = np.where(gs > gf_, gf_ / (gs + 1e-10), gs / (gf_ + 1e-10))
        AAF = 1 - np.abs(als - alf_) / (np.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    QAF = _q_pair(gA, aA, gF, aF)
    QBF = _q_pair(gB, aB, gF, aF)

    deno = np.sum(gA + gB)
    if deno < 1e-10:
        return 0.0
    return np.sum(QAF * gA + QBF * gB) / deno


def q_m(A, B, F, num_scales=2):
    """Q_M: Multiscale wavelet-based metric (Liu 2012, Eq.25).

    Haar wavelet decomposition, edge preservation at each scale.
    """
    def _haar(img):
        img = img.astype(np.float64)
        L = (img[:, 0::2] + img[:, 1::2]) / 2.0
        H = (img[:, 0::2] - img[:, 1::2]) / 2.0
        LL = (L[0::2, :] + L[1::2, :]) / 2.0
        LH = (L[0::2, :] - L[1::2, :]) / 2.0
        HL = (H[0::2, :] + H[1::2, :]) / 2.0
        HH = (H[0::2, :] - H[1::2, :]) / 2.0
        return LL, LH, HL, HH

    result = 1.0
    Ac, Bc, Fc = A.astype(np.float64), B.astype(np.float64), F.astype(np.float64)

    for s in range(num_scales):
        h, w = Ac.shape
        h, w = h - h % 2, w - w % 2
        Ac, Bc, Fc = Ac[:h, :w], Bc[:h, :w], Fc[:h, :w]

        LLa, LHa, HLa, HHa = _haar(Ac)
        LLb, LHb, HLb, HHb = _haar(Bc)
        LLf, LHf, HLf, HHf = _haar(Fc)

        def _ep(src, fused):
            return np.mean(np.exp(-np.abs(np.abs(src) - np.abs(fused))))

        ep_af = (_ep(LHa, LHf) + _ep(HLa, HLf) + _ep(HHa, HHf)) / 3.0
        ep_bf = (_ep(LHb, LHf) + _ep(HLb, HLf) + _ep(HHb, HHf)) / 3.0

        wA = np.sum(LHa ** 2 + HLa ** 2 + HHa ** 2)
        wB = np.sum(LHb ** 2 + HLb ** 2 + HHb ** 2)
        total = wA + wB + 1e-10

        q_s = (ep_af * wA + ep_bf * wB) / total
        result *= q_s ** (1.0 / num_scales)

        Ac, Bc, Fc = LLa, LLb, LLf

    return result


def q_sf(A, B, F):
    """Q_SF: Spatial Frequency ratio (Liu 2012, Eq.32).

    Q_SF = (SF_F - SF_R) / SF_R where SF_R uses max gradients from sources.
    """
    def _sf(img):
        img = img.astype(np.float64)
        m, n = img.shape
        RF = np.sum((img[:, 1:] - img[:, :-1]) ** 2) / (m * n)
        CF = np.sum((img[1:, :] - img[:-1, :]) ** 2) / (m * n)
        wd = 1.0 / np.sqrt(2.0)
        MDF = wd * np.sum((img[1:, 1:] - img[:-1, :-1]) ** 2) / (m * n)
        SDF = wd * np.sum((img[:-1, 1:] - img[1:, :-1]) ** 2) / (m * n)
        return np.sqrt(RF + CF + MDF + SDF)

    sf_f = _sf(F)

    # Reference SF from max gradients (Eq. 31)
    Af, Bf = A.astype(np.float64), B.astype(np.float64)
    m, n = Af.shape

    ref_H = np.maximum(np.abs(Af[:, 1:] - Af[:, :-1]), np.abs(Bf[:, 1:] - Bf[:, :-1]))
    ref_V = np.maximum(np.abs(Af[1:, :] - Af[:-1, :]), np.abs(Bf[1:, :] - Bf[:-1, :]))
    ref_MD = np.maximum(np.abs(Af[1:, 1:] - Af[:-1, :-1]), np.abs(Bf[1:, 1:] - Bf[:-1, :-1]))
    ref_SD = np.maximum(np.abs(Af[:-1, 1:] - Af[1:, :-1]), np.abs(Bf[:-1, 1:] - Bf[1:, :-1]))

    wd = 1.0 / np.sqrt(2.0)
    RF_r = np.sum(ref_H ** 2) / (m * n)
    CF_r = np.sum(ref_V ** 2) / (m * n)
    MDF_r = wd * np.sum(ref_MD ** 2) / (m * n)
    SDF_r = wd * np.sum(ref_SD ** 2) / (m * n)
    sf_r = np.sqrt(RF_r + CF_r + MDF_r + SDF_r)

    if sf_r < 1e-10:
        return 0.0
    return (sf_f - sf_r) / sf_r


def q_p(A, B, F):
    """Q_P: Phase Congruency-based metric (Liu 2012, Eq.33).

    Uses gradient magnitude (max moment), Laplacian (min moment),
    and gradient orientation (phase) as feature proxies.
    """
    Af, Bf, Ff = A.astype(np.float64), B.astype(np.float64), F.astype(np.float64)
    win = 7

    def _corr_map(x, y, ws):
        C = 1e-7
        mu_x = uniform_filter(x, size=ws)
        mu_y = uniform_filter(y, size=ws)
        sxy = uniform_filter(x * y, size=ws) - mu_x * mu_y
        sx = np.sqrt(np.maximum(uniform_filter(x ** 2, size=ws) - mu_x ** 2, 0))
        sy = np.sqrt(np.maximum(uniform_filter(y ** 2, size=ws) - mu_y ** 2, 0))
        return (sxy + C) / (sx * sy + C)

    gA, aA = _sobel_edge(Af)
    gB, aB = _sobel_edge(Bf)
    gF, aF = _sobel_edge(Ff)

    lapA = np.abs(cv2.Laplacian(Af, cv2.CV_64F))
    lapB = np.abs(cv2.Laplacian(Bf, cv2.CV_64F))
    lapF = np.abs(cv2.Laplacian(Ff, cv2.CV_64F))

    phA = (aA + np.pi) / (2 * np.pi)
    phB = (aB + np.pi) / (2 * np.pi)
    phF = (aF + np.pi) / (2 * np.pi)

    def _max_corr(fA, fB, fF):
        cAF = np.mean(_corr_map(fA, fF, win))
        cBF = np.mean(_corr_map(fB, fF, win))
        cSF = np.mean(_corr_map(fA + fB, fF, win))
        return max(cAF, cBF, cSF, 0)

    Pp = _max_corr(phA, phB, phF)
    PM = _max_corr(gA, gB, gF)
    Pm = _max_corr(lapA, lapB, lapF)

    return Pp * PM * Pm


# ============================================================
# GROUP 3: Structural Similarity-Based Metrics (Liu 2012)
# ============================================================

def q_s(A, B, F, win_size=8):
    """Q_S: Piella's Metric (Liu 2012, Eq.39).

    Saliency-weighted UIQI combination.
    """
    Af, Bf, Ff = A.astype(np.float64), B.astype(np.float64), F.astype(np.float64)
    q_af = _uiqi_map(Af, Ff, win_size)
    q_bf = _uiqi_map(Bf, Ff, win_size)
    sA = _local_variance(Af, win_size)
    sB = _local_variance(Bf, win_size)
    lam = sA / (sA + sB + 1e-10)
    return np.mean(lam * q_af + (1 - lam) * q_bf)


def q_c(A, B, F, win_size=8):
    """Q_C: Cvejic's Metric (Liu 2012, Eq.44).

    Covariance-based similarity weighting.
    """
    Af, Bf, Ff = A.astype(np.float64), B.astype(np.float64), F.astype(np.float64)
    q_af = _uiqi_map(Af, Ff, win_size)
    q_bf = _uiqi_map(Bf, Ff, win_size)

    sigma_af = uniform_filter(Af * Ff, size=win_size) - \
               uniform_filter(Af, size=win_size) * uniform_filter(Ff, size=win_size)
    sigma_bf = uniform_filter(Bf * Ff, size=win_size) - \
               uniform_filter(Bf, size=win_size) * uniform_filter(Ff, size=win_size)

    ratio = sigma_af / (sigma_af + sigma_bf + 1e-10)
    sim = np.clip(ratio, 0, 1)
    sim = np.where(ratio < 0, 0.0, sim)

    return np.mean(sim * q_af + (1 - sim) * q_bf)


def q_y(A, B, F, win_size=7):
    """Q_Y: Yang's Metric (Liu 2012, Eq.46).

    SSIM-based with threshold rule on SSIM(A,B).
    """
    Af, Bf, Ff = A.astype(np.float64), B.astype(np.float64), F.astype(np.float64)

    def _ssim_map_local(x, y, ws=7):
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu_x = uniform_filter(x, size=ws)
        mu_y = uniform_filter(y, size=ws)
        sxx = np.maximum(uniform_filter(x * x, size=ws) - mu_x ** 2, 0)
        syy = np.maximum(uniform_filter(y * y, size=ws) - mu_y ** 2, 0)
        sxy = uniform_filter(x * y, size=ws) - mu_x * mu_y
        return ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))

    ssim_ab = _ssim_map_local(Af, Bf, win_size)
    ssim_af = _ssim_map_local(Af, Ff, win_size)
    ssim_bf = _ssim_map_local(Bf, Ff, win_size)

    sA = _local_variance(Af, win_size)
    sB = _local_variance(Bf, win_size)
    lam = sA / (sA + sB + 1e-10)

    q_map = np.where(
        ssim_ab >= 0.75,
        lam * ssim_af + (1 - lam) * ssim_bf,
        np.maximum(ssim_af, ssim_bf)
    )
    return np.mean(q_map)


# ============================================================
# GROUP 4: Human Perception Inspired Metrics (Liu 2012)
# Standardized with VIFB metricsQcv.m and metricsQcb.m
# ============================================================

def q_cv(A, B, F):
    """Q_CV: Chen-Varshney Metric (Liu 2012, Eq.48).

    Standardized with VIFB metricsQcv.m:
    Mannos-Skarison CSF filter, 16x16 block, alpha=5.
    NOTE: Lower = better (measures distortion). We return negative for consistency.
    """
    im1 = _normalize255(A).astype(np.float64)
    im2 = _normalize255(B).astype(np.float64)
    fused = _normalize255(F).astype(np.float64)

    hang, lie = im1.shape
    windowSize = 16
    alpha = 5

    # Edge extraction (Sobel)
    gA, _ = _sobel_edge(im1)
    gB, _ = _sobel_edge(im2)

    # Local region saliency: sum of edge^alpha in each block
    H = hang // windowSize
    L = lie // windowSize

    ramda1 = np.zeros((H, L))
    ramda2 = np.zeros((H, L))
    for i in range(H):
        for j in range(L):
            block_a = gA[i * windowSize:(i + 1) * windowSize,
                         j * windowSize:(j + 1) * windowSize]
            block_b = gB[i * windowSize:(i + 1) * windowSize,
                         j * windowSize:(j + 1) * windowSize]
            ramda1[i, j] = np.sum(block_a ** alpha)
            ramda2[i, j] = np.sum(block_b ** alpha)

    # Difference images
    f1 = im1 - fused
    f2 = im2 - fused

    # CSF filter: Mannos-Skarison in frequency domain
    u_freq, v_freq = np.meshgrid(
        np.linspace(-1, 1, lie) * (lie / 8),
        np.linspace(-1, 1, hang) * (hang / 8)
    )
    r = np.sqrt(u_freq ** 2 + v_freq ** 2)
    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-(0.144 * r) ** 1.1)

    # Filter in frequency domain
    Df1 = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(f1)) * theta_m)))
    Df2 = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(f2)) * theta_m)))

    # Local MSE in each block
    D1 = np.zeros((H, L))
    D2 = np.zeros((H, L))
    for i in range(H):
        for j in range(L):
            block1 = Df1[i * windowSize:(i + 1) * windowSize,
                         j * windowSize:(j + 1) * windowSize]
            block2 = Df2[i * windowSize:(i + 1) * windowSize,
                         j * windowSize:(j + 1) * windowSize]
            D1[i, j] = np.mean(block1 ** 2)
            D2[i, j] = np.mean(block2 ** 2)

    # Global quality (Eq. in VIFB): weighted distortion
    total_saliency = np.sum(ramda1 + ramda2)
    if total_saliency < 1e-10:
        return 0.0
    Q = np.sum(ramda1 * D1 + ramda2 * D2) / total_saliency

    # Return negative distortion so higher = better
    return -Q


def q_cb(A, B, F):
    """Q_CB: Chen-Blum Metric (Liu 2012, Eq.54).

    Standardized with VIFB metricsQcb.m:
    DoG filter (f0=15.3870, f1=1.3456, a=0.7622),
    Gaussian 31x31, contrast masking (k=1, h=1, p=3, q=2, Z=0.0001).
    """
    im1 = _normalize255(A).astype(np.float64) / 255.0
    im2 = _normalize255(B).astype(np.float64) / 255.0
    fused_img = _normalize255(F).astype(np.float64) / 255.0

    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622
    k, h_param, p, q_param, Z = 1, 1, 3, 2, 0.0001

    hang, lie = im1.shape

    # DoG filter in frequency domain
    HH = hang / 30.0
    LL = lie / 30.0
    u_freq = np.linspace(-1, 1, lie) * LL
    v_freq = np.linspace(-1, 1, hang) * HH
    U, V = np.meshgrid(u_freq, v_freq)
    r = np.sqrt(U ** 2 + V ** 2)
    Sd = np.exp(-(r / f0) ** 2) - a * np.exp(-(r / f1) ** 2)

    # CSF filtering
    def _csf_filter(img):
        return np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(img)) * Sd)))

    filt1 = _csf_filter(im1)
    filt2 = _csf_filter(im2)
    filtF = _csf_filter(fused_img)

    # Gaussian kernels (31x31, sigma=2 and sigma=4)
    G1 = _gaussian_kernel(31, 2)
    G2 = _gaussian_kernel(31, 4)

    # Local contrast
    def _contrast(g1, g2, img):
        buff1 = fftconvolve(img, g1, mode='same')
        buff2 = fftconvolve(img, g2, mode='same')
        return buff1 / (buff2 + 1e-10) - 1

    C1 = np.abs(_contrast(G1, G2, filt1))
    C2 = np.abs(_contrast(G1, G2, filt2))
    Cf = np.abs(_contrast(G1, G2, filtF))

    # Contrast masking (Eq. 51)
    C1P = (k * C1 ** p) / (h_param * C1 ** q_param + Z)
    C2P = (k * C2 ** p) / (h_param * C2 ** q_param + Z)
    CfP = (k * Cf ** p) / (h_param * Cf ** q_param + Z)

    # Information preservation (Eq. 53)
    mask1 = (C1P < CfP).astype(np.float64)
    Q1F = (C1P / (CfP + 1e-10)) * mask1 + (CfP / (C1P + 1e-10)) * (1 - mask1)

    mask2 = (C2P < CfP).astype(np.float64)
    Q2F = (C2P / (CfP + 1e-10)) * mask2 + (CfP / (C2P + 1e-10)) * (1 - mask2)

    # Saliency weighting
    ramda1 = C1P ** 2 / (C1P ** 2 + C2P ** 2 + 1e-10)
    ramda2 = C2P ** 2 / (C1P ** 2 + C2P ** 2 + 1e-10)

    Q = ramda1 * Q1F + ramda2 * Q2F
    return np.mean(Q)


# ============================================================
# VIFB Additional Metrics
# ============================================================

def metric_en(A, B, F):
    """EN: Entropy of fused image (VIFB metricsEntropy.m)."""
    return _entropy_log2(F)


def metric_mi(A, B, F):
    """MI: Mutual Information (VIFB metricsMutinf.m).

    MI = MI(A,F) + MI(B,F) using natural log.
    """
    return _mutual_info_ln(A, F) + _mutual_info_ln(B, F)


def metric_sf(A, B, F):
    """SF: Spatial Frequency of fused image (VIFB metricsSpatial_frequency.m).

    SF = sqrt(RF + CF), only horizontal and vertical.
    """
    fused = F.astype(np.float64)
    m, n = fused.shape
    RF = np.sum((fused[:, 1:] - fused[:, :-1]) ** 2) / (m * n)
    CF = np.sum((fused[1:, :] - fused[:-1, :]) ** 2) / (m * n)
    return np.sqrt(RF + CF)


def metric_ag(A, B, F):
    """AG: Average Gradient (VIFB metricsAvg_gradient.m)."""
    fused = F.astype(np.float64)
    gy, gx = np.gradient(fused)
    s = np.sqrt((gx ** 2 + gy ** 2) / 2.0)
    r, c = fused.shape
    return np.sum(s) / ((r - 1) * (c - 1))


def metric_ei(A, B, F):
    """EI: Edge Intensity — mean Sobel gradient magnitude."""
    g, _ = _sobel_edge(F.astype(np.float64))
    return np.mean(g)


def metric_ssim(A, B, F):
    """SSIM: Structural Similarity (VIFB metricsSsim.m).

    Result = SSIM(A,F) + SSIM(B,F)  (sum, not average).
    """
    return _ssim_single(A, F) + _ssim_single(B, F)


def metric_ce(A, B, F):
    """CE: Cross Entropy (VIFB metricsCross_entropy.m).

    LOWER is better. CE = [CE(A,F) + CE(B,F)] / 2.
    """
    def _cross_entropy(img1, fused):
        img1 = img1.astype(np.float64)
        fused = fused.astype(np.float64)
        m, n = img1.shape
        h1 = np.bincount(img1.ravel().astype(np.int32).clip(0, 255), minlength=256).astype(np.float64)
        h2 = np.bincount(fused.ravel().astype(np.int32).clip(0, 255), minlength=256).astype(np.float64)
        P1 = h1 / (m * n)
        P2 = h2 / (m * n)
        mask = (P1 > 0) & (P2 > 0)
        return np.sum(P1[mask] * np.log2(P1[mask] / P2[mask]))

    return (_cross_entropy(A, F) + _cross_entropy(B, F)) / 2.0


def metric_psnr(A, B, F):
    """PSNR: Peak Signal-to-Noise Ratio (VIFB metricsPsnr.m).

    PSNR = avg of PSNR(A,F) and PSNR(B,F). Higher = better.
    """
    def _psnr(ref, fused):
        ref = ref.astype(np.float64)
        fused = fused.astype(np.float64)
        mse = np.mean((ref - fused) ** 2)
        if mse < 1e-10:
            return 100.0
        return 20.0 * np.log10(255.0 / np.sqrt(mse))

    return (_psnr(A, F) + _psnr(B, F)) / 2.0


def metric_mean(A, B, F):
    """Mean: Cường độ sáng trung bình của ảnh fusion (VIFB metricsVariance.m style)."""
    return np.mean(F.astype(np.float64))


def metric_sd(A, B, F):
    """SD: Độ lệch chuẩn (Standard Deviation) — đo độ tương phản/phương sai.

    SD = sqrt(sum((F - mean(F))^2) / (M*N))  (VIFB metricsVariance.m).
    """
    f = F.astype(np.float64)
    mu = np.mean(f)
    m, n = f.shape
    return np.sqrt(np.sum((f - mu) ** 2) / (m * n))


# ============================================================
# FusionMamba Paper Additional Metrics
# ============================================================

def metric_vif(A, B, F):
    """VIF: Visual Information Fidelity (sewar library)."""
    try:
        from sewar import vifp
        v1 = vifp(A, F)
        v2 = vifp(B, F)
        return (v1 + v2) / 2.0
    except ImportError:
        return float('nan')


def metric_scd(A, B, F):
    """SCD: Sum of Correlations of Differences."""
    def _corr(a, b):
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        a, b = a - a.mean(), b - b.mean()
        return np.dot(a, b) / (np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-10)

    f, s1, s2 = F.astype(np.float64), A.astype(np.float64), B.astype(np.float64)
    return _corr(f - s2, s1) + _corr(f - s1, s2)


def metric_ms_ssim(A, B, F):
    """MS-SSIM: Multi-Scale SSIM (sewar library)."""
    try:
        from sewar import msssim
        v1 = np.real(msssim(A, F))
        v2 = np.real(msssim(B, F))
        return (v1 + v2) / 2.0
    except ImportError:
        return float('nan')


def metric_fmi(A, B, F):
    """FMI: Feature Mutual Information (gradient-based)."""
    def _grad(img):
        img = img.astype(np.float64)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    def _nmi(a, b, bins=256):
        a_f, b_f = a.ravel(), b.ravel()
        a_min, a_max = a_f.min(), a_f.max()
        b_min, b_max = b_f.min(), b_f.max()
        if a_max - a_min < 1e-10:
            a_q = np.zeros_like(a_f, dtype=np.int32)
        else:
            a_q = ((a_f - a_min) / (a_max - a_min) * (bins - 1)).astype(np.int32)
        if b_max - b_min < 1e-10:
            b_q = np.zeros_like(b_f, dtype=np.int32)
        else:
            b_q = ((b_f - b_min) / (b_max - b_min) * (bins - 1)).astype(np.int32)

        hab = np.zeros((bins, bins), dtype=np.float64)
        np.add.at(hab, (a_q, b_q), 1)
        pxy = hab / hab.sum()
        px, py = pxy.sum(axis=1), pxy.sum(axis=0)
        hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
        hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / (px[:, None] * py[None, :])[nzs] + 1e-15))
        return mi / (max(hx, hy) + 1e-10)

    g1, g2, gf = _grad(A), _grad(B), _grad(F)
    return (_nmi(g1, gf) + _nmi(g2, gf)) / 2.0


# ============================================================
# Metric groups
# ============================================================

# ── Nhóm A: FusionMamba metrics — Similarity & Feature (4 metrics) ──
# Xie et al. (2024) FusionMamba — 4 metrics chính của paper
FUSIONMAMBA_METRICS = [
    ('MS_SSIM', 'Similarity',  metric_ms_ssim),  # Wang et al. 2003
    ('FMI',     'Similarity',  metric_fmi),       # Haghighat et al. 2011
    ('VIF',     'Fidelity',    metric_vif),       # Sheikh & Bovik 2006
    ('SCD',     'Fidelity',    metric_scd),       # Aslantas & Bendes 2015
]

# ── Nhóm B: Liu et al. 2012 — 12 metrics, 4 nhóm ──
LIU2012_METRICS = [
    # Group 1: Information Theory
    ('Q_MI',   'Info Theory',  q_mi),
    ('Q_TE',   'Info Theory',  q_te),
    ('Q_NCIE', 'Info Theory',  q_ncie),
    # Group 2: Image Feature
    ('Q_G',    'Img Feature',  q_g),
    ('Q_M',    'Img Feature',  q_m),
    ('Q_SF',   'Img Feature',  q_sf),
    ('Q_P',    'Img Feature',  q_p),
    # Group 3: Structural Similarity
    ('Q_S',    'Struct Sim',   q_s),
    ('Q_C',    'Struct Sim',   q_c),
    ('Q_Y',    'Struct Sim',   q_y),
    # Group 4: Human Perception
    ('Q_CV',   'Perception',   q_cv),
    ('Q_CB',   'Perception',   q_cb),
]

# ── Supplementary: EN + PSNR ──
SUPPLEMENTARY_METRICS = [
    ('EN',   'Supplementary', metric_en),       # Shannon entropy
    ('PSNR', 'Supplementary', metric_psnr),     # avg(PSNR(A,F), PSNR(B,F))
]

# ── Benchmark bắt buộc: 4 FusionMamba + 12 Liu2012 + EN + PSNR = 18 metrics ──
BENCHMARK_METRICS = FUSIONMAMBA_METRICS + LIU2012_METRICS + SUPPLEMENTARY_METRICS

# ── Nhóm phụ: VIFB (tùy chọn thêm) ──
VIFB_EXTRA = [
    ('MI',   'VIFB Extra',     metric_mi),
    ('SF',   'VIFB Extra',     metric_sf),
    ('EI',   'VIFB Extra',     metric_ei),
    ('SSIM', 'VIFB Extra',     metric_ssim),
    ('CE',   'VIFB Extra (↓)', metric_ce),
    ('PSNR', 'VIFB Extra',     metric_psnr),
]

# ── Nhóm cũ: Basic Quality (tham khảo) ──
BASIC_METRICS = [
    ('Mean', 'Basic Quality',  metric_mean),
    ('SD',   'Basic Quality',  metric_sd),
    ('AG',   'Basic Quality',  metric_ag),
    ('EN',   'Basic Quality',  metric_en),
]

ALL_METRICS = BENCHMARK_METRICS + VIFB_EXTRA + BASIC_METRICS


# ============================================================
# Evaluate
# ============================================================

def _rgb_to_ycbcr(img):
    """Convert BGR (OpenCV) to YCbCr. Returns Y, Cb, Cr as float64 [0,255]."""
    if len(img.shape) == 2:
        # Already grayscale — treat as Y, no color info
        return img.astype(np.float64), None, None
    # OpenCV reads as BGR
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)
    Cr = ycrcb[:, :, 1].astype(np.float64)
    Cb = ycrcb[:, :, 2].astype(np.float64)
    return Y, Cb, Cr


def evaluate_pair(src1_path, src2_path, fused_path, metrics_list=None, color_space='ycbcr'):
    """Evaluate a single image pair.

    color_space:
      'ycbcr' — Convert to YCbCr, compute metrics on Y (structure),
                report Cb/Cr average as color preservation metrics.
      'gray'  — Convert to grayscale (legacy behavior).
    """
    A_raw = cv2.imread(src1_path, cv2.IMREAD_UNCHANGED)
    B_raw = cv2.imread(src2_path, cv2.IMREAD_UNCHANGED)
    F_raw = cv2.imread(fused_path, cv2.IMREAD_UNCHANGED)

    if A_raw is None or B_raw is None or F_raw is None:
        return None

    # Resize sources to match fused
    h, w = F_raw.shape[:2]
    A_raw = cv2.resize(A_raw, (w, h))
    B_raw = cv2.resize(B_raw, (w, h))

    if metrics_list is None:
        metrics_list = ALL_METRICS

    if color_space == 'ycbcr':
        # Convert to YCbCr
        A_Y, A_Cb, A_Cr = _rgb_to_ycbcr(A_raw)
        B_Y, B_Cb, B_Cr = _rgb_to_ycbcr(B_raw)
        F_Y, F_Cb, F_Cr = _rgb_to_ycbcr(F_raw)

        has_color = (F_Cb is not None)

        # Check if all 3 images have color channels for Cb/Cr computation
        all_have_color = (A_Cb is not None) and (B_Cb is not None) and (F_Cb is not None)

        results = {}
        for name, group, func in metrics_list:
            # Compute on Y channel (structure)
            try:
                val_y = func(A_Y.astype(np.uint8), B_Y.astype(np.uint8), F_Y.astype(np.uint8))
                results[name] = val_y
            except Exception as e:
                print(f"    Warning: {name} (Y) failed: {e}")
                results[name] = float('nan')

            # Compute on Cb/Cr (color) only if ALL images have color channels
            if all_have_color:
                try:
                    val_cb = func(A_Cb.astype(np.uint8), B_Cb.astype(np.uint8), F_Cb.astype(np.uint8))
                    val_cr = func(A_Cr.astype(np.uint8), B_Cr.astype(np.uint8), F_Cr.astype(np.uint8))
                    results[name + '_Cb'] = val_cb
                    results[name + '_Cr'] = val_cr
                except Exception as e:
                    print(f"    Warning: {name} (Cb/Cr) failed: {e}")
                    results[name + '_Cb'] = float('nan')
                    results[name + '_Cr'] = float('nan')
        return results
    else:
        # Legacy grayscale mode
        A = cv2.cvtColor(A_raw, cv2.COLOR_BGR2GRAY) if len(A_raw.shape) == 3 else A_raw
        B = cv2.cvtColor(B_raw, cv2.COLOR_BGR2GRAY) if len(B_raw.shape) == 3 else B_raw
        F = cv2.cvtColor(F_raw, cv2.COLOR_BGR2GRAY) if len(F_raw.shape) == 3 else F_raw

        results = {}
        for name, group, func in metrics_list:
            try:
                results[name] = func(A, B, F)
            except Exception as e:
                print(f"    Warning: {name} failed: {e}")
                results[name] = float('nan')
        return results


def evaluate_all(source1_dir, source2_dir, fused_dir, metrics_list=None, color_space='ycbcr'):
    """Evaluate all fused images."""
    if metrics_list is None:
        metrics_list = ALL_METRICS

    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    fused_files = sorted([f for f in os.listdir(fused_dir) if f.lower().endswith(valid_ext)])

    if not fused_files:
        print(f"No images found in {fused_dir}")
        return None

    metric_names = [m[0] for m in metrics_list]
    all_results = []

    # Detect if color images (check first fused file)
    first_img = cv2.imread(os.path.join(fused_dir, fused_files[0]), cv2.IMREAD_UNCHANGED)
    has_color = (color_space == 'ycbcr') and (first_img is not None) and (len(first_img.shape) == 3)

    if has_color:
        # Build column names: Y channel + Cb + Cr
        display_names_y = [n + '(Y)' for n in metric_names]
        display_names_cb = [n + '(Cb)' for n in metric_names]
        display_names_cr = [n + '(Cr)' for n in metric_names]
    else:
        display_names_y = metric_names

    # ── Print Y channel table ──
    col_w = max(8, max(len(n) for n in display_names_y) + 1)
    header = f"{'Image':<18}" + "".join(f" {n:>{col_w}}" for n in display_names_y)
    cs_label = "YCbCr (Y channel)" if has_color else "Grayscale"
    print(f"\nEvaluating {len(fused_files)} images | Color space: {cs_label}\n")
    print(header)
    print("-" * len(header))

    for fname in fused_files:
        src1_path = os.path.join(source1_dir, fname)
        src2_path = os.path.join(source2_dir, fname)
        fused_path = os.path.join(fused_dir, fname)

        # Try alternate extensions
        for sp, sd in [(src1_path, source1_dir), (src2_path, source2_dir)]:
            if not os.path.exists(sp):
                base = os.path.splitext(fname)[0]
                for ext in valid_ext:
                    cand = os.path.join(sd, base + ext)
                    if os.path.exists(cand):
                        if sd == source1_dir:
                            src1_path = cand
                        else:
                            src2_path = cand
                        break

        if not os.path.exists(src1_path) or not os.path.exists(src2_path):
            print(f"  {fname:<18} SKIPPED (source not found)")
            continue

        result = evaluate_pair(src1_path, src2_path, fused_path, metrics_list, color_space)
        if result:
            all_results.append(result)
            row = f"  {fname:<18}"
            for name in metric_names:
                row += f" {result[name]:>{col_w}.4f}"
            print(row)

    if not all_results:
        return None

    print("-" * len(header))
    avg_y = {k: np.nanmean([r[k] for r in all_results]) for k in metric_names}

    row = f"  {'AVERAGE':<18}"
    for name in metric_names:
        row += f" {avg_y[name]:>{col_w}.4f}"
    print(row)

    # ── Summary: Y channel ──
    print(f"\n{'=' * 60}")
    print(f"  Summary ({len(all_results)} images) — Y channel (Structure)")
    print(f"{'=' * 60}")
    current_group = ""
    for name, group, _ in metrics_list:
        if group != current_group:
            current_group = group
            print(f"\n  [{group}]")
        print(f"    {name:<12} {avg_y[name]:>10.4f}")
    print(f"{'=' * 60}")

    # ── Summary: Cb/Cr channels (color preservation) ──
    if has_color:
        avg_cb = {k: np.nanmean([r.get(k + '_Cb', float('nan')) for r in all_results]) for k in metric_names}
        avg_cr = {k: np.nanmean([r.get(k + '_Cr', float('nan')) for r in all_results]) for k in metric_names}

        print(f"\n{'=' * 60}")
        print(f"  Color Preservation — Cb / Cr channels")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<12} {'Cb':>10} {'Cr':>10} {'Avg(Cb,Cr)':>12}")
        print(f"  {'-'*46}")
        for name in metric_names:
            cb_val = avg_cb[name]
            cr_val = avg_cr[name]
            avg_color = (cb_val + cr_val) / 2.0
            print(f"  {name:<12} {cb_val:>10.4f} {cr_val:>10.4f} {avg_color:>12.4f}")
        print(f"{'=' * 60}")

    return avg_y


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image Fusion Benchmark Metrics')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI'])
    parser.add_argument('--fused_dir', type=str, default=None)
    parser.add_argument('--source1_dir', type=str, default=None)
    parser.add_argument('--source2_dir', type=str, default=None)
    parser.add_argument('--group', type=str, default='benchmark',
                        choices=['benchmark', 'basic', 'liu2012', 'all'],
                        help='benchmark=16 bắt buộc (basic+liu2012), all=26 metrics')
    parser.add_argument('--color', type=str, default='ycbcr',
                        choices=['ycbcr', 'gray'],
                        help='ycbcr=đo trên Y/Cb/Cr (mặc định), gray=convert grayscale')
    args = parser.parse_args()

    DATA_ROOT = '/data2/anhvv/ai_nlp_rs/datasets'
    if args.dataset:
        modality = args.dataset.split('-')[0]
        args.fused_dir = args.fused_dir or f'./outputs/{args.dataset}'
        args.source1_dir = args.source1_dir or f'{DATA_ROOT}/data_{args.dataset}/test/{modality}'
        args.source2_dir = args.source2_dir or f'{DATA_ROOT}/data_{args.dataset}/test/MRI'

    if not args.fused_dir or not args.source1_dir or not args.source2_dir:
        parser.error("Specify --dataset or all three dirs")

    # Select metrics
    if args.group == 'benchmark':
        metrics = BENCHMARK_METRICS
    elif args.group == 'basic':
        metrics = BASIC_METRICS
    elif args.group == 'liu2012':
        metrics = LIU2012_METRICS
    else:
        metrics = ALL_METRICS

    print(f"Fused:   {args.fused_dir}")
    print(f"Source1: {args.source1_dir}")
    print(f"Source2: {args.source2_dir}")
    print(f"Group:   {args.group} ({len(metrics)} metrics)")
    print(f"Color:   {args.color}")

    evaluate_all(args.source1_dir, args.source2_dir, args.fused_dir, metrics, args.color)
