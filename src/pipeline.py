# src/pipeline.py

"""
Pipeline utama FP KK Early Warning Hoax.

Fungsi utama:
- run_llm_fuzzy_pipeline(...)

Alur:
1. Load dataset mentah dari CSV.
2. Hitung 3 skor LLM (intensitas_emosi, kecurigaan_format, kredibilitas_rendah).
3. Hitung resiko_hoaks dengan fuzzy.
4. (Opsional) Simpan hasil ke file processed.
"""

from __future__ import annotations
from typing import Optional
import os
import pandas as pd
from src.config import DATA_RAW_PATH, DATA_LLM_PATH
from src.llm_features import hitung_skor_llm_df
from src.fuzzy_system import hitung_resiko_hoaks


def run_llm_fuzzy_pipeline(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    text_col: str = "text",
    save_result: bool = False,
) -> pd.DataFrame:
    """
    Jalankan seluruh pipeline LLM + Fuzzy.

    Param:
    - input_path : path CSV mentah (kalau None, pakai config.DATA_RAW_PATH)
    - output_path: path CSV hasil (kalau None, pakai config.DATA_LLM_PATH)
    - text_col   : nama kolom teks di dataset
    - save_result: kalau True → hasil disimpan ke output_path

    Return:
    - DataFrame hasil dengan kolom:
      [text_col, intensitas_emosi, kecurigaan_format, kredibilitas_rendah, resiko_hoaks, ...kolom lain...]
    """
    if input_path is None:
        input_path = DATA_RAW_PATH
    if output_path is None:
        output_path = DATA_LLM_PATH

    # 1) Load dataset mentah
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    df_raw = pd.read_csv(input_path)

    if text_col not in df_raw.columns:
        raise KeyError(
            f"Kolom teks '{text_col}' tidak ada di dataset. "
            f"Kolom yang tersedia: {list(df_raw.columns)}"
        )

    # 2) Hitung skor LLM
    df_llm = hitung_skor_llm_df(df_raw, text_col=text_col)

    # 3) Hitung resiko_hoaks fuzzy
    def _hitung_resiko_row(row):
        return hitung_resiko_hoaks(
            intensitas_emosi=row["intensitas_emosi"],
            kecurigaan_format=row["kecurigaan_format"],
            kredibilitas_rendah=row["kredibilitas_rendah"],
            output_scale_100=True,
        )

    df_llm["resiko_hoaks"] = df_llm.apply(_hitung_resiko_row, axis=1)

    # 4) Simpan kalau diminta
    if save_result:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_llm.to_csv(output_path, index=False)
        print(f"[INFO] Hasil pipeline disimpan ke: {output_path}")

    return df_llm
