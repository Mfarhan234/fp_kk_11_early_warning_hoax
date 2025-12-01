# src/fuzzy_system.py

"""
Fuzzy Inference System (Mamdani sederhana) untuk resiko hoaks.

Input (semua 0–1, dari LLM):
- intensitas_emosi
- kecurigaan_format
- kredibilitas_rendah

Output:
- resiko_hoaks dalam skala 0–1 (bisa dikali 100 kalau mau %)

Catatan:
- Membership function pakai bentuk segitiga / shoulder.
- Inferensi Mamdani dengan aturan IF–THEN.
- Defuzzifikasi pakai centroid (numerik, sampling 0..1).
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple
import numpy as np


# =========================
# 1. Membership Functions
# =========================

def tri_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular / shoulder membership function yang tahan kasus a==b atau b==c.

    a <= b <= c

    - Kalau a < b < c → segitiga biasa
    - Kalau a == b   → left-shoulder (nilai 1 di [a,b], turun ke 0 di c)
    - Kalau b == c   → right-shoulder (naik dari 0 di a ke 1 di [b,c])
    """
    x = float(x)

    # Segitiga biasa
    if a < b < c:
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)

    # Left shoulder: a == b < c
    if a == b < c:
        if x <= a:
            return 1.0
        elif x >= c:
            return 0.0
        else:
            return (c - x) / (c - a)

    # Right shoulder: a < b == c
    if a < b == c:
        if x <= a:
            return 0.0
        elif x >= c:
            return 1.0
        else:
            return (x - a) / (b - a)

    # Degenerat (semua sama)
    if a == b == c:
        return 1.0 if x == a else 0.0

    return 0.0


# Definisi membership untuk tiap variabel input (0–1)

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
    """
    kredibilitas_rendah kecil → sumber cukup kredibel.
    """
    return tri_mf(x, 0.0, 0.0, 0.4)


def mu_kredrendah_sedang(x: float) -> float:
    return tri_mf(x, 0.2, 0.5, 0.8)


def mu_kredrendah_tinggi(x: float) -> float:
    """
    kredibilitas_rendah tinggi → sumber sangat tidak kredibel.
    """
    return tri_mf(x, 0.6, 1.0, 1.0)


# Output resiko_hoaks (0–1)
def mu_resiko_rendah(y: float) -> float:
    return tri_mf(y, 0.0, 0.0, 0.4)


def mu_resiko_sedang(y: float) -> float:
    return tri_mf(y, 0.2, 0.5, 0.8)


def mu_resiko_tinggi(y: float) -> float:
    return tri_mf(y, 0.6, 1.0, 1.0)


# =========================
# 2. Aturan Fuzzy
# =========================

"""
Kita definisikan aturan dalam bentuk:

(alpha, label_output)

dimana alpha = min(derajat membership kondisi)

Contoh ide aturan (bisa kamu tweak nanti):
1. Jika emosi TINGGI & format TINGGI & kredibilitas_rendah TINGGI → resiko TINGGI
2. Jika format TINGGI & kredibilitas_rendah SEDANG/ TINGGI → resiko TINGGI
3. Jika emosi SEDANG & format SEDANG & kredibilitas_rendah SEDANG → resiko SEDANG
4. Jika format RENDAH & kredibilitas_rendah RENDAH → resiko RENDAH
5. Jika emosi RENDAH & format RENDAH tapi kredibilitas_rendah TINGGI → resiko SEDANG (karena sumber meragukan)
"""


def _hitung_rule_strengths(
    intensitas_emosi: float,
    kecurigaan_format: float,
    kredibilitas_rendah: float,
) -> List[Tuple[float, str]]:
    e = float(intensitas_emosi)
    f = float(kecurigaan_format)
    k = float(kredibilitas_rendah)

    # derajat membership
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

    # R6: emosi tinggi tapi format rendah (emosi dari isi, bukan format):
    #     jika kred_rendah sedang/tinggi → resiko sedang
    alpha6 = min(e_H, max(k_M, k_H))
    rules.append((alpha6, "sedang"))

    # R7: emosi sedang & format tinggi & kred_rendah rendah → resiko sedang
    alpha7 = min(e_M, f_H, k_L)
    rules.append((alpha7, "sedang"))

    return rules


# mapping label → fungsi membership output
OUTPUT_MFS: Dict[str, Callable[[float], float]] = {
    "rendah": mu_resiko_rendah,
    "sedang": mu_resiko_sedang,
    "tinggi": mu_resiko_tinggi,
}


# =========================
# 3. Defuzzifikasi Mamdani
# =========================

def _defuzz_centroid_0_1(
    rules: List[Tuple[float, str]],
    n_points: int = 201,
) -> float:
    """
    Defuzzifikasi dengan metode centroid pada domain 0–1 (sampling).

    rules: list (alpha, label_output)
    """
    ys = np.linspace(0.0, 1.0, n_points)
    mu_agg = np.zeros_like(ys)

    for alpha, label in rules:
        if alpha <= 0.0:
            continue
        mf = OUTPUT_MFS[label]
        # ambil membership output yang sudah dipotong alpha (min)
        vals = np.array([min(alpha, mf(y)) for y in ys])
        mu_agg = np.maximum(mu_agg, vals)

    denom = mu_agg.sum()
    if denom == 0.0:
        # tidak ada firing → anggap resiko 0
        return 0.0

    num = (ys * mu_agg).sum()
    return float(num / denom)


# =========================
# 4. Fungsi publik
# =========================

def hitung_resiko_hoaks(
    intensitas_emosi: float,
    kecurigaan_format: float,
    kredibilitas_rendah: float,
    output_scale_100: bool = True,
) -> float:
    """
    Fungsi utama yang akan dipakai di mana-mana.

    Input:
    - intensitas_emosi     : 0–1
    - kecurigaan_format    : 0–1
    - kredibilitas_rendah  : 0–1

    Output:
    - jika output_scale_100=True → 0–100
    - kalau False → 0–1
    """
    # clamp input ke 0–1
    e = max(0.0, min(1.0, float(intensitas_emosi)))
    f = max(0.0, min(1.0, float(kecurigaan_format)))
    k = max(0.0, min(1.0, float(kredibilitas_rendah)))

    rules = _hitung_rule_strengths(e, f, k)
    resiko_0_1 = _defuzz_centroid_0_1(rules)

    if output_scale_100:
        return resiko_0_1 * 100.0
    return resiko_0_1
