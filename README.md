# Early Warning Hoax – Fuzzy + Metaheuristik (PSO & GWO)

Repositori ini berisi implementasi sistem **early warning hoax** untuk konten media sosial.Input yang digunakan:

- **Teks** postingan
- **Gambar postingan**
- **Foto profil akun**

Tiga sumber informasi tersebut dinilai oleh **LLM (Gemini)** untuk menghasilkan tiga skor numerik, kemudian diproses oleh:

- **Fuzzy Mamdani** sebagai model utama penilaian risiko hoaks
- **PSO (Particle Swarm Optimization)** dan **GWO (Grey Wolf Optimizer)** sebagai metode metaheuristik untuk mengeksplorasi bobot fitur (opsional, untuk eksperimen).

Aplikasi demo dapat dijalankan menggunakan **Streamlit**.

---

## 1. Struktur Proyek

Struktur folder utama (diringkas):

```text
fp_kk_11_early_warning_hoax/
├─ app/
│  ├─ __init__.py
│  └─ app_streamlit.py        # UI Streamlit
│
├─ data/
│  ├─ image/
│  │  ├─ postingan/           # Gambar postingan (postingan_1.png, dst.)
│  │  └─ profil/              # Gambar profil (profil_1.png, dst.)
│  ├─ processed/
│  │  ├─ hoax_dataset_llm.csv
│  │  └─ hoax_dataset_llm_multimodal.csv
│  └─ raw/
│     ├─ hoax_dataset_raw_converted.csv
│     └─ hoax_dataset_multimodal.csv
│
├─ notebooks/
│  ├─ 00_original_experiment.ipynb
│  ├─ 01_eksplorasi_dataset.ipynb
│  ├─ 02_llm_scoring.ipynb
│  └─ 03_llm_fuzzy_pipeline.ipynb
│
├─ src/
│  ├─ __init__.py
│  ├─ config.py               # Konfigurasi & loader .env
│  ├─ llm_features.py         # Pemanggilan Gemini & ekstraksi skor LLM
│  ├─ fuzzy_system.py         # Definisi FIS Mamdani (fungsi keanggotaan & inferensi)
│  ├─ pso_optimizer.py        # Optimisasi bobot fitur dengan PSO
│  ├─ gwo_optimizer.py        # Optimisasi bobot fitur dengan GWO
│  ├─ pipeline.py             # Utility pipeline (menghubungkan LLM → Fuzzy → Optimizer)
│  └─ viz.py                  # Fungsi visualisasi (grafik fuzzy / distribusi skor)
│
├─ .env                       # Menyimpan GOOGLE_API_KEY (tidak di-commit)
├─ requirements.txt
└─ README.md
```

## 2. Persiapan Lingkungan

Disarankan menggunakan **Python 3.12** (Python 3.14 saat ini masih banyak paket yang belum stabil).

### 2.1. Clone & pindah ke folder proyek

```bash
git clone <URL_REPO_KAMU>.git
cd fp_kk_11_early_warning_hoax
```

### 2.2. Buat virtual environment

Windows (pakai Python launcher):

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

> Pastikan di terminal muncul `(.venv)` di awal baris.

### 2.3. Upgrade pip dan install requirements

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Paket utama di `requirements.txt` antara lain:

* `google-generativeai` – akses Gemini API
* `python-dotenv` – load `.env`
* `pandas`, `numpy` – olah data
* `scikit-learn` – utilitas numerik / scaling / dsb.
* `matplotlib`, `seaborn` – visualisasi
* `Pillow` – load gambar
* `streamlit` – web UI sederhana

---

## 3. Konfigurasi API Key (Gemini)

Buat file `.env` di root proyek:

```env
GOOGLE_API_KEY=ISI_API_KEY_GEMINI_DI_SINI
```

Di dalam kode, API key diambil lewat `src/config.py`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
```

Tanpa `.env` yang benar, modul `llm_features.py` tidak bisa memanggil Gemini.

---

## 4. Dataset & Format Data

### 4.1. Data mentah (`data/raw/`)

* `hoax_dataset_raw_converted.csv`

  Format sederhana:

  ```csv
  id,text
  1,"PAK JOKOWI RUNTUHKAN RENCANA ..."
  2,"✨* * Prabowo Minta Kejagung Usut ..."
  ...
  ```
* `hoax_dataset_multimodal.csv`

  Sudah dilengkapi kolom file gambar:

  ```csv
  id,text,file_postingan,file_profil
  1,"SEGERA SEBARKAN!!! ...","postingan_1.png","profil_1.png"
  2,"Pemerintah mengumumkan ...","postingan_2.png","profil_2.png"
  ...
  ```

  Gambar fisik berada di:

  * `data/image/postingan/postingan_X.png`
  * `data/image/profil/profil_X.png`

### 4.2. Data hasil LLM (`data/processed/`)

Setelah pemanggilan Gemini, dataset diperkaya dengan tiga skor:

* `intensitas_emosi` – seberapa emosional / provokatif konten
* `kecurigaan_format` – seberapa mencurigakan dari sisi format (typography, tanda baca, clickbait, dsb.)
* `kredibilitas_rendah` – indikasi rendahnya kredibilitas sumber

Contoh (ringkas):

```csv
id,text,file_postingan,file_profil,intensitas_emosi,kecurigaan_format,kredibilitas_rendah
1,"PAK JOKOWI ...",postingan_1.png,profil_1.png,0.9,0.9,0.8
...
```

File:

* `hoax_dataset_llm.csv` – untuk teks saja
* `hoax_dataset_llm_multimodal.csv` – untuk teks + gambar

---

## 5. Alur Logika Sistem

Secara garis besar, pipeline bekerja seperti ini:

1. **Input (Teks + Gambar)**

   Dari dataset atau dari input pengguna (Streamlit):

   * Teks postingan
   * Gambar postingan
   * Foto profil akun
2. **Ekstraksi Skor oleh LLM (Gemini)** – `src/llm_features.py`

   * Gambar & teks dikirim ke Gemini (model misalnya `gemini-2.5-flash`).
   * Model diminta mengembalikan **3 skor numerik** antara 0–1:
     * `intensitas_emosi`
     * `kecurigaan_format`
     * `kredibilitas_rendah`
   * Terdapat fungsi batch seperti `hitung_skor_llm_df_multimodal(df, ...)` yang:
     * Loop setiap baris dataset
     * Membuka gambar dengan `PIL.Image.open`
     * Memanggil Gemini
     * Menyusun DataFrame hasil dan menyimpannya ke `data/processed/…csv`
3. **Fuzzy Inference System (Mamdani)** – `src/fuzzy_system.py`

   * Mendifinisikan fungsi keanggotaan untuk masing-masing input:
     * **Emosi:** rendah / sedang / tinggi
     * **Format:** aman / mencurigakan / sangat mencurigakan
     * **Kredibilitas:** tinggi / sedang / rendah
   * Aturan fuzzy (rule base) menggabungkan tiga input tersebut menjadi output `resiko_hoaks` (0–100) dengan kategori:
     * rendah
     * sedang
     * tinggi
   * Proses:
     1. Fuzzifikasi skor 0–1
     2. Evaluasi aturan (min / max)
     3. Agregasi output
     4. Defuzzifikasi (mis. centroid) → **skor numerik** 0–100
4. **PSO & GWO (Opsional, Metaheuristik)** – `src/pso_optimizer.py`, `src/gwo_optimizer.py`

   * Digunakan untuk **mencari bobot** untuk tiga skor LLM:

     ( w = [w_emosi, w_format, w_kredibilitas] )
   * Secara konseptual:

     * PSO/GWO mengeksplorasi kombinasi bobot
     * Setiap kombinasi bobot dipakai untuk men-skala input sebelum masuk ke Fuzzy
     * Fungsi objektif bisa disesuaikan, misalnya:
       * Meng-upscale contoh yang kelihatan ekstrem,
       * Menjaga distribusi skor tertentu,
       * Atau menyesuaikan ke label manual (jika di masa depan ada anotasi).
   * Hasil akhirnya berupa **bobot terbaik** yang kemudian bisa dipakai saat inference.
5. **Visualisasi** – `src/viz.py`

   * Menampilkan kurva fungsi keanggotaan output (`rendah`, `sedang`, `tinggi`)
   * Menandai hasil defuzzifikasi sebagai garis vertikal pada grafik
   * Digunakan baik di notebook maupun di Streamlit (misal grafik yang sudah terlihat di demo).
6. **Aplikasi Streamlit** – `app/app_streamlit.py`

   * Menyediakan UI interaktif:
     * Input manual tiga skor fuzzy (0–1) via slider **atau**
     * (rencana pengembangan) input:
       * teks,
       * gambar postingan,
       * foto profil.
   * Pengguna bisa memilih mode:
     * **Fuzzy baseline** (tanpa PSO)
     * (opsional) **Fuzzy + PSO** / **Fuzzy + GWO** jika sudah diintegrasikan
   * Menampilkan:
     * Nilai skor `resiko_hoaks`
     * Kategori (Rendah / Sedang / Tinggi)
     * Grafik fungsi keanggotaan output.

---

## 6. Menjalankan Pipeline (Notebook)

Untuk keperluan eksperimen / dokumentasi, alur bisa diikuti lewat notebook:

1. **Eksperimen awal & definisi fuzzy**

   Buka `notebooks/00_original_experiment.ipynb`

   – berisi uji coba awal fungsi keanggotaan, rule base, dan grafik.
2. **Eksplorasi dataset**

   `notebooks/01_eksplorasi_dataset.ipynb`

   – EDA terhadap `data/raw/hoax_dataset_raw_converted.csv`.
3. **Hitung skor LLM (Gemini)**

   `notebooks/02_llm_scoring.ipynb`

   – Membaca `hoax_dataset_multimodal.csv`, memanggil Gemini, dan menyimpan:

   * `data/processed/hoax_dataset_llm_multimodal.csv`
4. **Fuzzy + Visualisasi + (Opsional) Optimizer**

   `notebooks/03_llm_fuzzy_pipeline.ipynb`

   – Membaca file processed hasil Gemini

   – Menghitung `resiko_hoaks` dengan Fuzzy

   – Visualisasi distribusi skor

   – Penggunaan bobot hasil PSO/GWO (jika diaktifkan).

---

## 7. Menjalankan Aplikasi Streamlit

Pastikan:

* Virtual env aktif (`(.venv)` muncul di terminal)
* Dependencies sudah terinstall
* File `.env` berisi `GOOGLE_API_KEY` (untuk fitur LLM jika diintegrasikan ke UI)

Jalankan:

```bash
python -m streamlit run app/app_streamlit.py
```

Output di terminal akan menampilkan URL lokal, misalnya:

```text
Local URL: http://localhost:8501
Network URL: http://192.168.0.100:8501
```

Buka salah satu URL tersebut di browser.

### Fitur di halaman Streamlit (versi saat ini)

* Slider input:
  * `intensitas_emosi` (0–1)
  * `kecurigaan_format` (0–1)
  * `kredibilitas_rendah` (0–1)
* Tombol **Hitung Resiko Hoaks**
* Panel hasil:
  * Skor risiko hoaks (0–100)
  * Kategori (Rendah/Sedang/Tinggi)
* Grafik:
  * Kurva fungsi keanggotaan output (`rendah`, `sedang`, `tinggi`)
  * Garis vertikal hasil defuzzifikasi

> Ke depan, UI dapat diperluas untuk:
>
> * Upload teks + gambar (postingan & profil),
> * Menjalankan pipeline LLM → Fuzzy secara end-to-end,
> * Memilih mode Fuzzy / Fuzzy+PSO / Fuzzy+GWO dari sidebar.

---

## 8. Cara Menggunakan Kode di `src/` Secara Programatik

Contoh singkat pemakaian modul di luar Streamlit / notebook:

```python
from pathlib import Path
import pandas as pd

from src.llm_features import hitung_skor_llm_df_multimodal
from src.fuzzy_system import hitung_resiko_hoaks

# 1. Baca data multimodal mentah
df_raw = pd.read_csv("data/raw/hoax_dataset_multimodal.csv")

# 2. Hitung skor LLM
df_llm = hitung_skor_llm_df_multimodal(
    df_raw,
    text_col="text",
    file_post_col="file_postingan",
    file_profil_col="file_profil",
)

# 3. Hitung skor fuzzy per baris
df_llm["resiko_hoaks"] = df_llm.apply(
    lambda row: hitung_resiko_hoaks(
        intensitas_emosi=float(row["intensitas_emosi"]),
        kecurigaan_format=float(row["kecurigaan_format"]),
        kredibilitas_rendah=float(row["kredibilitas_rendah"]),
        output_scale_100=True,
    ),
    axis=1,
)

df_llm.to_csv("data/processed/hoax_dataset_llm_multimodal.csv", index=False)
```

Jika bobot dari PSO/GWO sudah tersedia:

```python
from src.pso_optimizer import w_best_pso

def pred_with_pso(row):
    emosi = float(row["intensitas_emosi"]) * w_best_pso[0]
    fmt   = float(row["kecurigaan_format"]) * w_best_pso[1]
    kred  = float(row["kredibilitas_rendah"]) * w_best_pso[2]

    return hitung_resiko_hoaks(
        intensitas_emosi=emosi,
        kecurigaan_format=fmt,
        kredibilitas_rendah=kred,
        output_scale_100=True,
    )

df_llm["resiko_hoaks_pso"] = df_llm.apply(pred_with_pso, axis=1)
```

---

## 9. Catatan & Batasan

* Dataset saat ini **belum memiliki label ground truth** (hoaks / bukan hoaks).

  Fuzzy + PSO/GWO lebih berperan sebagai **sistem scoring risiko** dibanding klasifikasi biner.
* Kualitas skor sangat bergantung pada:

  * Prompt ke Gemini,
  * Desain fungsi keanggotaan fuzzy,
  * Aturan fuzzy yang disusun.
* PSO & GWO di sini digunakan sebagai **contoh penerapan metaheuristik** untuk mengatur bobot fitur, bukan sebagai klaim bahwa ini solusi optimal final.

---
