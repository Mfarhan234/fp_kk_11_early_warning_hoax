"""
Fuzzy Inference System (Mamdani) untuk menghitung resiko_hoaks
berdasarkan 3 skor LLM (0–1):
- intensitas_emosi
- kecurigaan_format
- kredibilitas_rendah
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple
import numpy as np
from .pso_optimizer import BEST_W_PSO

# 1. Membership Function
def tri_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular / shoulder membership function.
    a <= b <= c
    - a < b < c : segitiga biasa
    - a == b    : left shoulder
    - b == c    : right shoulder
    """
    x = float(x)

    # Segitiga biasa
    if a < b < c:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)

    # Left shoulder: a == b < c
    if a == b < c:
        if x <= a:
            return 1.0
        if x >= c:
            return 0.0
        return (c - x) / (c - a)

    # Right shoulder: a < b == c
    if a < b == c:
        if x <= a:
            return 0.0
        if x >= c:
            return 1.0
        return (x - a) / (b - a)

    # Degenerate: a == b == c
    if a == b == c:
        return 1.0 if x == a else 0.0

    return 0.0


# Membership variabel input (0–1) 
def mu_emosi_rendah(x: float) -> float:
    return tri_mf(x, 0.0, 0.0, 0.4)

def mu_emosi_sedang(x: float) -> float:
    return tri_mf(x, 0.2, 0.5, 0.8)

def mu_emosi_tinggi(x: float) -> float:
    return tri_mf(x, 0.6, 1.0, 1.0)

def mu_format_rendah(x: float) -> float:
    return tri_mf(x, 0.0, 0.0, 0.4)

def mu_format_sedang(x: float) -> float:
    return tri_mf(x, 0.2, 0.5, 0.8)

def mu_format_tinggi(x: float) -> float:
    return tri_mf(x, 0.6, 1.0, 1.0)

def mu_kredrendah_rendah(x: float) -> float:
    """kredibilitas_rendah kecil → sumber cenderung kredibel."""
    return tri_mf(x, 0.0, 0.0, 0.4)

def mu_kredrendah_sedang(x: float) -> float:
    return tri_mf(x, 0.2, 0.5, 0.8)

def mu_kredrendah_tinggi(x: float) -> float:
    """kredibilitas_rendah tinggi → sumber sangat tidak kredibel."""
    return tri_mf(x, 0.6, 1.0, 1.0)


# Membership output resiko_hoaks (0–1) 
def mu_resiko_rendah(y: float) -> float:
    return tri_mf(y, 0.0, 0.0, 0.4)

def mu_resiko_sedang(y: float) -> float:
    return tri_mf(y, 0.2, 0.5, 0.8)

def mu_resiko_tinggi(y: float) -> float:
    return tri_mf(y, 0.6, 1.0, 1.0)

# Aturan Fuzzy
def _hitung_rule_strengths(
    intensitas_emosi: float,
    kecurigaan_format: float,
    kredibilitas_rendah: float,
) -> List[Tuple[float, str]]:
    """
    Hitung firing strength tiap rule.
    Return list (alpha, label_output) dengan label: 'rendah' / 'sedang' / 'tinggi'.
    """
    e = float(intensitas_emosi)
    f = float(kecurigaan_format)
    k = float(kredibilitas_rendah)

    # derajat membership input
    e_L = mu_emosi_rendah(e)
    e_M = mu_emosi_sedang(e)
    e_H = mu_emosi_tinggi(e)

    f_L = mu_format_rendah(f)
    f_M = mu_format_sedang(f)
    f_H = mu_format_tinggi(f)

    k_L = mu_kredrendah_rendah(k)
    k_M = mu_kredrendah_sedang(k)
    k_H = mu_kredrendah_tinggi(k)

    rules: List[Tuple[float, str]] = []

    # R1: emosi tinggi & format tinggi & kred_rendah tinggi → resiko tinggi
    alpha1 = min(e_H, f_H, k_H)
    rules.append((alpha1, "tinggi"))

    # R2: format tinggi & kred_rendah sedang/tinggi → resiko tinggi
    alpha2 = min(f_H, max(k_M, k_H))
    rules.append((alpha2, "tinggi"))

    # R3: emosi sedang & format sedang & kred_rendah sedang → resiko sedang
    alpha3 = min(e_M, f_M, k_M)
    rules.append((alpha3, "sedang"))

    # R4: format rendah & kred_rendah rendah → resiko rendah
    alpha4 = min(f_L, k_L)
    rules.append((alpha4, "rendah"))

    # R5: emosi rendah & format rendah & kred_rendah tinggi → resiko sedang
    alpha5 = min(e_L, f_L, k_H)
    rules.append((alpha5, "sedang"))

    # R6: emosi tinggi, kred_rendah sedang/tinggi (emosi dari isi, bukan format) → sedang
    alpha6 = min(e_H, max(k_M, k_H))
    rules.append((alpha6, "sedang"))

    # R7: emosi sedang & format tinggi & kred_rendah rendah → resiko sedang
    alpha7 = min(e_M, f_H, k_L)
    rules.append((alpha7, "sedang"))

    return rules

# mapping label → membership output
OUTPUT_MFS: Dict[str, Callable[[float], float]] = {
    "rendah": mu_resiko_rendah,
    "sedang": mu_resiko_sedang,
    "tinggi": mu_resiko_tinggi,
}

# Defuzzifikasi
def _defuzz_centroid_0_1(rules: List[Tuple[float, str]], n_points: int = 201,) -> float:
    """
    Defuzzifikasi centroid pada domain [0, 1] dengan sampling.
    rules: list (alpha, label_output).
    """
    ys = np.linspace(0.0, 1.0, n_points)
    mu_agg = np.zeros_like(ys)

    for alpha, label in rules:
        if alpha <= 0.0:
            continue
        mf = OUTPUT_MFS[label]
        # output fuzzy dipotong alpha (Mamdani: min)
        vals = np.array([min(alpha, mf(y)) for y in ys])
        mu_agg = np.maximum(mu_agg, vals)

    denom = mu_agg.sum()
    if denom == 0.0:
        return 0.0

    num = (ys * mu_agg).sum()
    return float(num / denom)


# Fungsi publik
def hitung_resiko_hoaks(
    intensitas_emosi: float,
    kecurigaan_format: float,
    kredibilitas_rendah: float,
    output_scale_100: bool = True,
) -> float:
    """
    Fungsi utama fuzzy Mamdani.
    Input  0–1 → output 0–1 atau 0–100 (kalau output_scale_100=True).
    """
    # Clamp input ke [0, 1]
    e = max(0.0, min(1.0, float(intensitas_emosi)))
    f = max(0.0, min(1.0, float(kecurigaan_format)))
    k = max(0.0, min(1.0, float(kredibilitas_rendah)))

    rules = _hitung_rule_strengths(e, f, k)
    resiko_0_1 = _defuzz_centroid_0_1(rules)

    if output_scale_100:
        return resiko_0_1 * 100.0
    return resiko_0_1

def hitung_resiko_hoaks_pso(
    intensitas_emosi: float,
    kecurigaan_format: float,
    kredibilitas_rendah: float,
    weights=None,
    output_scale_100: bool = True,
) -> float:
    """
    Wrapper fuzzy + bobot PSO.
    - Kalau weights=None → pakai BEST_W_PSO dari pso_optimizer.py
    - Kalau weights diisi array/list 3 elemen → pakai itu.
    """
    if weights is None:
        w = np.array(BEST_W_PSO, dtype=float)
    else:
        w = np.array(weights, dtype=float)

    emo = np.clip(intensitas_emosi * w[0], 0.0, 1.0)
    frm = np.clip(kecurigaan_format * w[1], 0.0, 1.0)
    kred = np.clip(kredibilitas_rendah * w[2], 0.0, 1.0)

    return hitung_resiko_hoaks(
        intensitas_emosi=emo,
        kecurigaan_format=frm,
        kredibilitas_rendah=kred,
        output_scale_100=output_scale_100,
    )
