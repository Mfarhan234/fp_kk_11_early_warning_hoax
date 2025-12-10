"""
Konfigurasi global proyek FP KK Early Warning Hoax.
- Membaca GOOGLE_API_KEY dari file .env di root project.
- Menyimpan path default dataset.
"""
# src/config.py (potongan)
import os
from pathlib import Path
from dotenv import load_dotenv

# muat .env dari root project
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# ... variabel config lain tetap ...


# PATH DATASET 
DATA_RAW_PATH = "data/raw/hoax_dataset_raw.csv"
DATA_LLM_PATH = "data/processed/hoax_dataset_llm.csv"
