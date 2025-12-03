# app/app_streamlit.py
import io
from typing import Dict, Any
import os
import sys

# === Tambah: pastikan root project ke sys.path ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Import modul lokal
from src.fuzzy_system import (
    hitung_resiko_hoaks,
    hitung_resiko_hoaks_pso,
)
from src.llm_features import (
    get_gemini_model,
    PROMPT_INSTRUKSI,
    _parse_scores_from_text,
)

# Bobot PSO (fallback ke [1,1,1] kalau belum didefinisikan di modul)
try:
    from src.pso_optimizer import BEST_W_PSO  # definisikan di pso_optimizer.py
except Exception:
    BEST_W_PSO = np.array([1.0, 1.0, 1.0])


# =====================================================================
# Helper: panggil Gemini multimodal untuk 1 kasus (teks + 0/1/2 gambar)
# =====================================================================
def hitung_skor_llm_single(
    teks: str,
    file_postingan,
    file_profil,
) -> Dict[str, float]:
    """
    Mengembalikan dict:
    {
      "intensitas_emosi": float 0-1,
      "kecurigaan_format": float 0-1,
      "kredibilitas_rendah": float 0-1
    }
    """
    model = get_gemini_model("gemini-2.5-flash")

    parts: list[Any] = [
        PROMPT_INSTRUKSI,
        "\n\nTEKS YANG HARUS DINILAI:\n\n",
        teks or "",
    ]

    if file_postingan is not None:
        img_post = Image.open(file_postingan).convert("RGB")
        parts.append("\n\n[Gambar Postingan]")
        parts.append(img_post)

    if file_profil is not None:
        img_prof = Image.open(file_profil).convert("RGB")
        parts.append("\n\n[Gambar Profil]")
        parts.append(img_prof)

    resp = model.generate_content(parts)
    raw_text = resp.text or ""
    scores = _parse_scores_from_text(raw_text)
    return scores


# =====================================================================
# Helper: kategorisasi & rekomendasi
# =====================================================================
def kategori_resiko(score: float) -> str:
    if score < 33:
        return "Rendah"
    elif score < 66:
        return "Sedang"
    return "Tinggi"


def rekomendasi_tindakan(score: float) -> str:
    k = kategori_resiko(score)
    if k == "Rendah":
        return (
            "📗 Resiko hoaks rendah. Konten cenderung aman, "
            "tetap boleh dibagikan namun tetap kritis pada sumber."
        )
    if k == "Sedang":
        return (
            "🟠 Resiko hoaks sedang. Disarankan cek fakta dulu "
            "(misal ke media resmi / cekfakta) sebelum ikut menyebarkan."
        )
    return (
        "🚨 Resiko hoaks tinggi. Sebaiknya *jangan* disebarkan, "
        "laporkan ke pihak terkait bila perlu dan berikan edukasi ke orang sekitar."
    )


# =====================================================================
# Helper: plot membership fungsi output hoax_risk
# =====================================================================
def plot_fuzzy_output(score: float, title: str = "Fuzzy Output: resiko_hoaks"):
    x = np.linspace(0, 100, 500)

    def low(x):
        return np.clip((40 - x) / 40, 0, 1)

    def medium(x):
        return np.maximum(
            np.minimum((x - 20) / (50 - 20), (80 - x) / (80 - 50)),
            0,
        )

    def high(x):
        return np.clip((x - 60) / (100 - 60), 0, 1)

    y_low = low(x)
    y_med = medium(x)
    y_high = high(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_low, label="rendah")
    ax.plot(x, y_med, label="sedang")
    ax.plot(x, y_high, label="tinggi")

    ax.fill_between(x, 0, y_low, alpha=0.1)
    ax.fill_between(x, 0, y_med, alpha=0.1)
    ax.fill_between(x, 0, y_high, alpha=0.1)

    ax.axvline(score, color="k")

    ax.set_title(title)
    ax.set_xlabel("resiko_hoaks (0–100)")
    ax.set_ylabel("Membership")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


# =====================================================================
# Streamlit App
# =====================================================================
st.set_page_config(
    page_title="Early Warning Hoax – Fuzzy + PSO",
    layout="wide",
)

st.sidebar.title("Input Konten")

# Mode tampilan (hanya Fuzzy & PSO)
mode = st.sidebar.radio("Lihat hasil:", ["Fuzzy", "PSO"])

# Input multimodal
teks_input = st.sidebar.text_area(
    "Teks / caption postingan",
    height=150,
    placeholder="Tempel teks postingan di sini...",
)

file_postingan = st.sidebar.file_uploader(
    "Gambar postingan (opsional)",
    type=["png", "jpg", "jpeg"],
)

file_profil = st.sidebar.file_uploader(
    "Foto profil (opsional)",
    type=["png", "jpg", "jpeg"],
)

hitung_btn = st.sidebar.button("🔍 Analisis Konten")

# state untuk menyimpan hasil terakhir
if "hasil_analisis" not in st.session_state:
    st.session_state.hasil_analisis = None

if hitung_btn:
    if not teks_input and file_postingan is None and file_profil is None:
        st.sidebar.warning("Minimal isi teks atau upload salah satu gambar.")
    else:
        with st.spinner("Memanggil Gemini & menghitung fuzzy..."):
            scores = hitung_skor_llm_single(
                teks=teks_input,
                file_postingan=file_postingan,
                file_profil=file_profil,
            )

            emosi = float(scores["intensitas_emosi"])
            fmt = float(scores["kecurigaan_format"])
            kred = float(scores["kredibilitas_rendah"])

            # Fuzzy baseline
            skor_fuzzy = hitung_resiko_hoaks(
                intensitas_emosi=emosi,
                kecurigaan_format=fmt,
                kredibilitas_rendah=kred,
                output_scale_100=True,
            )

            # Fuzzy + PSO (menggunakan BEST_W_PSO)
            skor_pso = hitung_resiko_hoaks_pso(
                intensitas_emosi=emosi,
                kecurigaan_format=fmt,
                kredibilitas_rendah=kred,
                weights=BEST_W_PSO,
                output_scale_100=True,
            )

        st.session_state.hasil_analisis = {
            "scores": scores,
            "skor_fuzzy": skor_fuzzy,
            "skor_pso": skor_pso,
        }

# ============================
# Tampilan utama
# ============================
st.title("🧠 Early Warning Hoax – Demo Fuzzy + PSO")

st.markdown(
    """
Aplikasi ini menerima **teks**, **gambar postingan**, dan **foto profil**,
kemudian:

1. Gemini memberi 3 skor dasar (0–1):
   - `intensitas_emosi`
   - `kecurigaan_format`
   - `kredibilitas_rendah`
2. Skor tersebut masuk ke **Fuzzy Mamdani** → `resiko_hoaks` (0–100).
3. Mode **PSO** menunjukkan variasi skor ketika input terlebih dulu
   diskalakan dengan bobot hasil optimasi.
"""
)

hasil = st.session_state.hasil_analisis

if hasil is None:
    st.info("Masukkan teks/gambar di sidebar lalu tekan **Analisis Konten**.")
    st.stop()

scores = hasil["scores"]
skor_fuzzy = hasil["skor_fuzzy"]
skor_pso = hasil["skor_pso"]

# ──────────────────────────────────────────────
# Layout 2 kolom
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

# ====================== KIRI: ringkasan skor =========================
with col_left:
    if mode == "Fuzzy":
        st.subheader("Hasil Fuzzy (Baseline)")

        st.metric("Skor Resiko Hoaks", f"{skor_fuzzy:.2f}")
        st.write("Kategori:", f"**{kategori_resiko(skor_fuzzy)}**")
        st.write(rekomendasi_tindakan(skor_fuzzy))

    elif mode == "PSO":
        st.subheader("Hasil Fuzzy + Bobot PSO")

        st.metric("Skor Resiko Hoaks (PSO)", f"{skor_pso:.2f}")
        st.write("Kategori:", f"**{kategori_resiko(skor_pso)}**")
        st.write(rekomendasi_tindakan(skor_pso))

        st.caption(
            "Bobot PSO yang dipakai adalah hasil training offline pada core dataset UMPO "
            f"(w = {np.round(BEST_W_PSO, 3)}). "
            "Bobot ini menskalakan intensitas_emosi, kecurigaan_format, dan kredibilitas_rendah "
            "sebelum masuk ke fuzzy Mamdani."
        )

    st.markdown("---")
    st.subheader("Ringkasan Skor dari Gemini (0–1)")
    st.write(f"- intensitas_emosi: `{scores['intensitas_emosi']:.2f}`")
    st.write(f"- kecurigaan_format: `{scores['kecurigaan_format']:.2f}`")
    st.write(f"- kredibilitas_rendah: `{scores['kredibilitas_rendah']:.2f}`")

# ====================== KANAN: visualisasi fuzzy =====================
with col_right:
    st.subheader("Visualisasi Fuzzy Output")

    if mode == "Fuzzy":
        fig = plot_fuzzy_output(skor_fuzzy, "Fuzzy Output – Baseline")
    else:  # PSO
        fig = plot_fuzzy_output(skor_pso, "Fuzzy Output – Input Terbobot PSO")

    st.pyplot(fig)
