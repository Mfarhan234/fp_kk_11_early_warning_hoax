# src/llm_features.py

"""
Modul fitur LLM (Gemini) untuk FP Early Warning Hoax.

- Menggunakan model: gemini-2.5-flash
- Output 3 skor 0–1:
    - intensitas_emosi
    - kecurigaan_format
    - kredibilitas_rendah
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image
import json
from typing import Any, Dict
import google.generativeai as genai
from src.config import GOOGLE_API_KEY
import pandas as pd

# ─── BASE PATH PROJECT ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
IMG_POSTINGAN_DIR = BASE_DIR / "data" / "image" / "postingan"
IMG_PROFIL_DIR    = BASE_DIR / "data" / "image" / "profil"



# ========== Inisialisasi model ==========

def get_gemini_model(model_name: str = "gemini-2.5-flash"):
    """
    Inisialisasi dan mengembalikan objek model Gemini.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY kosong. "
            "Cek file .env dan src/config.py."
        )

    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(model_name)


# ========== Prompt instruksi LLM ==========

PROMPT_INSTRUKSI = """Kamu adalah analis konten media sosial Indonesia yang bertugas menilai
seberapa kuat emosi, seberapa sensasional format, dan seberapa rendah
kredibilitas sumber sebuah postingan.

⚠ PENTING:
- Kamu TIDAK diminta memutuskan "hoaks atau bukan".
- Kamu HANYA memberi tiga skor dasar (0–1) yang nanti akan digunakan
  sistem lain untuk menghitung:
  - potensi_viral (seberapa mudah menyebar), dan
  - resiko_hoaks (seberapa berisiko mengandung hoaks).
- Jadi fokusmu adalah menilai tiga faktor dasar ini seobjektif mungkin.

==================================================
DEFINISI TIGA PARAMETER
==================================================

1) intensitas_emosi (0–1)
   Mengukur seberapa emosional isi postingan.

   0.0  ≈ sangat netral / informatif
   0.3  ≈ agak menyentuh, tapi masih tenang
   0.5  ≈ cukup emosional (sedih, khawatir, iba)
   0.8  ≈ sangat emosional (takut, panik, marah)
   1.0  ≈ sangat provokatif / menakut-nakuti / penuh kemarahan

   Contoh kalimat yang menaikkan intensitas_emosi:
   - banyak kata "musibah", "bencana besar", "azab", "bahaya", "ancaman",
     "korban berjatuhan", dsb.
   - nada panik, menyalahkan pihak tertentu, atau menghakimi.

2) kecurigaan_format (0–1)
   Mengukur seberapa sensasional / mirip "pesan berantai" format postingan.

   0.0  ≈ format sangat wajar: huruf biasa, tanda baca normal,
          tidak ada ajakan "sebar ke semua orang".
   0.3  ≈ ada sedikit dramatisasi (1–2 kata kapital, 1 tanda seru),
          tapi masih terasa seperti pengumuman normal.
   0.5  ≈ cukup sensasional: beberapa kata kapital, gaya bahasa hiperbolik.
   0.8  ≈ sangat sensasional: BANYAK HURUF KAPITAL, BANYAK !!!,
          judul heboh, gaya clickbait.
   1.0  ≈ pola klasik "forward pesan berantai" / "broadcast hoaks":
          penuh CAPS, !!!, dan ajakan menyebarkan.

   Hal-hal yang MENAIKKAN kecurigaan_format:
   - BANYAK huruf kapital (SELURUH KALIMAT DALAM CAPS).
   - Banyak tanda seru (!!!), tanda tanya (???), atau kombinasi aneh.
   - Kalimat seperti: "SEGERA SEBARKAN", "FORWARD KE SEMUA KONTAK",
     "JANGAN ABAIKAN PESAN INI", "VIRALKAN SEKARANG".

   Hal-hal yang MENURUNKAN kecurigaan_format:
   - Desain poster lembaga resmi dengan logo jelas, informasi rapi,
     gaya bahasa formal, dan TANPA ajakan menyebarkan pesan berantai.

3) kredibilitas_rendah (0–1)
   Mengukur seberapa TIDAK kredibel sumber / akun yang mengunggah.

   0.0  ≈ sangat kredibel:
          - lembaga pemerintah, media arus utama, NGO resmi,
          - tokoh publik yang dikenal luas,
          - informasi kontak dan identitas jelas.
   0.3  ≈ cenderung kredibel tapi tidak resmi (komunitas besar, organisasi yang dikenal).
   0.5  ≈ netral / meragukan (akun pribadi biasa, identitas tidak terlalu jelas).
   0.8  ≈ sangat meragukan (nama akun aneh, bio tidak jelas, tampak seperti akun spam).
   1.0  ≈ sangat tidak kredibel (akun baru, tidak jelas pemiliknya, konten penuh provokasi).

   Catatan penting:
   - Jika postingan berasal dari lembaga resmi atau tokoh publik terkenal
     (misalnya kementerian, lembaga penanggulangan bencana, media nasional,
      organisasi kemanusiaan resmi), maka kredibilitas_rendah seharusnya kecil
     (misal 0.0–0.3), kecuali ada indikasi kuat sebaliknya.

==================================================
KETERKAITAN DENGAN VIRAL & HOAKS
==================================================

Sistem lain akan menggunakan:
- intensitas_emosi + kecurigaan_format → untuk menghitung potensi_viral
- kecurigaan_format + kredibilitas_rendah → untuk menghitung resiko_hoaks

Tugasmu BUKAN menghitung viral/hoaks, tetapi memberikan nilai dasar yang konsisten:
- Konten bisa saja sangat viral tetapi resiko hoaks rendah
  (contoh: penggalangan dana resmi, kampanye kemanusiaan).
- Konten bisa saja tidak terlalu viral tetapi resiko hoaks tinggi
  (contoh: klaim medis palsu dari akun kecil).

==================================================
FORMAT JAWABAN
==================================================

Jawab SELALU dalam format JSON murni seperti ini:

{
  "intensitas_emosi": 0.8,
  "kecurigaan_format": 0.9,
  "kredibilitas_rendah": 0.7
}

- Gunakan titik (.) sebagai pemisah desimal.
- Jangan menambahkan teks lain di luar JSON.
- Jika ragu, gunakan nilai tengah (misal 0.3, 0.5, 0.7) daripada 0.0 atau 1.0 ekstrem.

==================================================
CONTOH PENILAIAN
==================================================

[Contoh 1 – Donasi resmi, emosional tapi tidak mencurigakan]
Caption ringkas: ajakan berdonasi untuk korban bencana melalui rekening
resmi lembaga besar; bahasa sopan, informatif, tanpa ajakan "sebar ke semua".
Gambar: poster donasi dengan logo lembaga resmi, nomor rekening bank nasional,
kontak resmi, dan desain rapi.

Jawaban yang wajar:
{
  "intensitas_emosi": 0.5,
  "kecurigaan_format": 0.2,
  "kredibilitas_rendah": 0.1
}

[Contoh 2 – Pesan berantai hoaks]
Caption ringkas: "SEGERA SEBARKAN!!! VIRUS BARU MENYERANG!! HANYA DENGAN
MINUM AIR GARAM ANDA AKAN SELAMAT!!! KIRIM KE SEMUA TEMAN ANDA!!!"
Gambar: screenshot chat WhatsApp.

Jawaban yang wajar:
{
  "intensitas_emosi": 0.9,
  "kecurigaan_format": 1.0,
  "kredibilitas_rendah": 0.8
}

[Contoh 3 – Berita serius dari media resmi]
Caption ringkas: laporan gempa bumi dari akun resmi BMKG atau media nasional,
bahasa formal, tanpa ajakan sebar berantai.
Gambar: infografis resmi atau foto lokasi bencana.

Jawaban yang wajar:
{
  "intensitas_emosi": 0.6,
  "kecurigaan_format": 0.2,
  "kredibilitas_rendah": 0.1
}

[Contoh 4 – Rumor dari akun pribadi meragukan]
Caption ringkas: klaim bahwa "pemerintah menutup-nutupi fakta tertentu"
tanpa sumber jelas, dengan nada menuduh, tapi tidak banyak tanda seru.
Gambar: foto buram atau tidak relevan.

Jawaban yang wajar:
{
  "intensitas_emosi": 0.7,
  "kecurigaan_format": 0.5,
  "kredibilitas_rendah": 0.7
}
"""

# ========== Helper parsing skor ==========

def _clean_score(val: Any, default: float = 0.0) -> float:
    """
    Konversi ke float dan clamp 0–1.
    """
    try:
        x = float(val)
    except Exception:
        return default

    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    return x


def _parse_scores_from_text(text: str) -> Dict[str, float]:
    """
    Baca JSON dari text LLM dan ekstrak 3 skor.

    Strategi:
    1) Coba json.loads langsung
    2) Kalau gagal, ambil substring di antara { ... } terakhir
    """
    text = text.strip()

    # 1. coba langsung
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 2. coba potong bagian { ... }
        if "{" in text and "}" in text:
            sub = text[text.find("{"): text.rfind("}") + 1]
            data = json.loads(sub)
        else:
            raise

    intens = _clean_score(data.get("intensitas_emosi", 0.0))
    kform = _clean_score(data.get("kecurigaan_format", 0.0))
    kred  = _clean_score(data.get("kredibilitas_rendah", 0.0))

    return {
        "intensitas_emosi": intens,
        "kecurigaan_format": kform,
        "kredibilitas_rendah": kred,
    }


# ========== Fungsi utama: 1 teks ==========

def skor_hoax_dari_teks(
    teks: str,
    model: Any | None = None,
) -> Dict[str, float]:
    """
    Hitung tiga skor LLM (0–1) untuk satu teks.

    Output dict:
    {
      "intensitas_emosi": float,
      "kecurigaan_format": float,
      "kredibilitas_rendah": float,
    }
    """
    if model is None:
        model = get_gemini_model()

    if teks is None:
        teks = ""
    teks = str(teks).strip()

    if not teks:
        return {
            "intensitas_emosi": 0.0,
            "kecurigaan_format": 0.0,
            "kredibilitas_rendah": 0.0,
        }

    prompt = PROMPT_INSTRUKSI + "\n\nTEKS YANG HARUS DINILAI:\n\n" + teks

    resp = model.generate_content(prompt)
    raw_text = resp.text or ""

    try:
        scores = _parse_scores_from_text(raw_text)
    except Exception as e:
        print("[WARNING] Gagal parse skor LLM, respons mentah:")
        print(raw_text)
        print("Error:", repr(e))
        scores = {
            "intensitas_emosi": 0.0,
            "kecurigaan_format": 0.0,
            "kredibilitas_rendah": 0.0,
        }

    return scores

def hitung_skor_llm_df(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Hitung skor LLM untuk setiap baris DataFrame.

    Param:
    - df       : DataFrame yang setidaknya punya kolom teks (text_col)
    - text_col : nama kolom teks (default 'text', ganti sesuai datasetmu)

    Return:
    - DataFrame copy dengan 3 kolom baru:
        'intensitas_emosi', 'kecurigaan_format', 'kredibilitas_rendah'
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Kolom '{text_col}' tidak ditemukan di DataFrame. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df_out = df.copy()
    model = get_gemini_model("gemini-2.5-flash")

    intens_list = []
    format_list = []
    kred_list = []

    for teks in df_out[text_col].fillna("").astype(str):
        skor = skor_hoax_dari_teks(teks, model=model)
        intens_list.append(skor["intensitas_emosi"])
        format_list.append(skor["kecurigaan_format"])
        kred_list.append(skor["kredibilitas_rendah"])

    df_out["intensitas_emosi"] = intens_list
    df_out["kecurigaan_format"] = format_list
    df_out["kredibilitas_rendah"] = kred_list

    return df_out

def skor_hoax_multimodal(
    teks: str,
    path_postingan: Path | None = None,
    path_profil: Path | None = None,
    model: Any | None = None,
) -> Dict[str, float]:
    """
    Hitung tiga skor LLM (0–1) untuk satu postingan dengan TEKS + (opsional) GAMBAR.
    """
    if model is None:
        model = get_gemini_model()

    if teks is None:
        teks = ""
    teks = str(teks).strip()

    # Kalau benar-benar tidak ada apa-apa
    if not teks and not path_postingan and not path_profil:
        return {
            "intensitas_emosi": 0.0,
            "kecurigaan_format": 0.0,
            "kredibilitas_rendah": 0.0,
        }

    # Susun konten multimodal
    content_parts: list[Any] = [
        PROMPT_INSTRUKSI,
        "\n\nTEKS YANG HARUS DINILAI:\n\n" + teks,
    ]

    # Tambahkan gambar (kalau ada)
    for p in (path_postingan, path_profil):
        if p is None:
            continue
        try:
            img = Image.open(p)
            content_parts.append(img)
        except FileNotFoundError:
            print(f"[WARNING] File gambar tidak ditemukan: {p}")
        except Exception as e:
            print(f"[WARNING] Gagal membuka gambar {p}: {e!r}")

    resp = model.generate_content(content_parts)
    raw_text = resp.text or ""

    try:
        scores = _parse_scores_from_text(raw_text)
    except Exception as e:
        print("[WARNING] Gagal parse skor LLM (multimodal), respons mentah:")
        print(raw_text)
        print("Error:", repr(e))
        scores = {
            "intensitas_emosi": 0.0,
            "kecurigaan_format": 0.0,
            "kredibilitas_rendah": 0.0,
        }

    return scores


def hitung_skor_llm_df_multimodal(
    df: pd.DataFrame,
    text_col: str = "text",
    file_post_col: str = "file_postingan",
    file_profil_col: str = "file_profil",
) -> pd.DataFrame:
    """
    Hitung skor LLM untuk setiap baris DataFrame dengan TEKS + GAMBAR.
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Kolom '{text_col}' tidak ditemukan di DataFrame. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df_out = df.copy()
    model = get_gemini_model("gemini-2.5-flash")

    intens_list: list[float] = []
    format_list: list[float] = []
    kred_list: list[float] = []

    for _, row in df_out.iterrows():
        teks = str(row.get(text_col, "") or "")

        fname_post = row.get(file_post_col) if file_post_col in df_out.columns else None
        fname_prof = row.get(file_profil_col) if file_profil_col in df_out.columns else None

        path_post = None
        path_prof = None

        if isinstance(fname_post, str) and fname_post.strip():
            path_post = IMG_POSTINGAN_DIR / fname_post.strip()

        if isinstance(fname_prof, str) and fname_prof.strip():
            path_prof = IMG_PROFIL_DIR / fname_prof.strip()

        skor = skor_hoax_multimodal(
            teks=teks,
            path_postingan=path_post,
            path_profil=path_prof,
            model=model,
        )

        intens_list.append(skor["intensitas_emosi"])
        format_list.append(skor["kecurigaan_format"])
        kred_list.append(skor["kredibilitas_rendah"])

    df_out["intensitas_emosi"] = intens_list
    df_out["kecurigaan_format"] = format_list
    df_out["kredibilitas_rendah"] = kred_list

    return df_out
