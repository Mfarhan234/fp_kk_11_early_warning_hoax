# src/config.py

"""
Konfigurasi global proyek FP KK Early Warning Hoax.

- Membaca GOOGLE_API_KEY dari file .env di root project.
- Menyimpan path default dataset (boleh kamu sesuaikan).
"""

import os
from dotenv import load_dotenv

# Baca .env di root project
load_dotenv()

# ========== API KEY ==========

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    print("[PERINGATAN] GOOGLE_API_KEY belum di-set di file .env")


# ========== PATH DATASET (BOLEH DISESUAIKAN) ==========

# contoh default, ganti kalau nama file kamu beda
DATA_RAW_PATH = "data/raw/hoax_dataset_raw.csv"
DATA_LLM_PATH = "data/processed/hoax_dataset_llm.csv"
