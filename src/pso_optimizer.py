# src/pso_optimizer.py

"""
Optimasi Fuzzy Input Scaling dengan Particle Swarm Optimization (PSO).

Ide:
- Input FIS: intensitas_emosi, kecurigaan_format, kredibilitas_rendah (0–1)
- Kita beri skala w = [w1, w2, w3] sehingga:
    emosi'  = clip(w1 * emosi,  0, 1)
    format' = clip(w2 * format, 0, 1)
    kred'   = clip(w3 * kred,   0, 1)
- Nilai w dicari dengan PSO agar MSE terhadap label ahli minimal.

Catatan:
- df harus punya kolom:
    'intensitas_emosi', 'kecurigaan_format', 'kredibilitas_rendah', dan target_col
- target_col misalnya: 'resiko_hoaks_ahli'
"""

from __future__ import annotations
from typing import Callable, Tuple, List, Dict
import numpy as np
import pandas as pd
from src.fuzzy_system import hitung_resiko_hoaks


# ----------------------------------------------------------------------
# PSO generik untuk vektor real-valued
# ----------------------------------------------------------------------


def pso_optimize(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    n_particles: int = 30,
    n_iters: int = 50,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int | None = None,
) -> Dict[str, object]:
    """
    PSO sederhana untuk meminimalkan objective_fn.

    Parameters
    ----------
    objective_fn : function
        Fungsi f(x) yang menerima vektor 1D (shape = [dim]) dan
        mengembalikan nilai fitness (semakin kecil semakin baik).
    dim : int
        Dimensi ruang pencarian.
    bounds : (lower, upper)
        lower, upper masing-masing np.ndarray shape [dim],
        berisi batas bawah dan atas untuk tiap dimensi.
    n_particles : int
        Jumlah partikel dalam swarm.
    n_iters : int
        Jumlah iterasi PSO.
    w, c1, c2 : float
        Parameter PSO:
        - w  : inertia
        - c1 : cognitive (tarikan ke pbest)
        - c2 : social (tarikan ke gbest)
    seed : int | None
        Seed random untuk reprodusibilitas.

    Returns
    -------
    dict dengan keys:
        - 'best_position' : np.ndarray shape [dim]
        - 'best_fitness'  : float
        - 'history'       : list[float] (best_fitness per iter)
    """
    rng = np.random.default_rng(seed)

    lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Inisialisasi posisi partikel (uniform di dalam bounds)
    positions = rng.uniform(lower, upper, size=(n_particles, dim))
    velocities = np.zeros_like(positions)

    # Evaluasi awal
    fitness = np.array([objective_fn(p) for p in positions])

    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    gbest_idx = np.argmin(fitness)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    history: List[float] = [float(gbest_fit)]

    for _ in range(n_iters):
        # Update velocity & position
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )

        positions = positions + velocities

        # Terapkan batas
        positions = np.clip(positions, lower, upper)

        # Hitung fitness baru
        fitness = np.array([objective_fn(p) for p in positions])

        # Update personal best
        improved = fitness < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness[improved]

        # Update global best
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < gbest_fit:
            gbest_fit = float(fitness[min_idx])
            gbest_pos = positions[min_idx].copy()

        history.append(gbest_fit)

    return {
        "best_position": gbest_pos,
        "best_fitness": float(gbest_fit),
        "history": history,
    }


# ----------------------------------------------------------------------
# Khusus: Optimasi scaling input FIS Mamdani
# ----------------------------------------------------------------------


def _build_fuzzy_scaling_objective(
    df: pd.DataFrame,
    target_col: str,
    bounds_scale: Tuple[float, float] = (0.5, 1.5),
) -> Tuple[Callable[[np.ndarray], float], int, Tuple[np.ndarray, np.ndarray]]:
    """
    Siapkan objective function PSO untuk mencari skala input fuzzy.

    df harus punya kolom:
        - 'intensitas_emosi'
        - 'kecurigaan_format'
        - 'kredibilitas_rendah'
        - target_col (label ahli, misal 'resiko_hoaks_ahli')
    """
    for col in ["intensitas_emosi", "kecurigaan_format", "kredibilitas_rendah", target_col]:
        if col not in df.columns:
            raise KeyError(
                f"Kolom '{col}' tidak ditemukan di DataFrame. "
                f"Kolom yang tersedia: {list(df.columns)}"
            )

    emo = df["intensitas_emosi"].astype(float).to_numpy()
    frm = df["kecurigaan_format"].astype(float).to_numpy()
    kred = df["kredibilitas_rendah"].astype(float).to_numpy()
    y_true = df[target_col].astype(float).to_numpy()

    dim = 3  # w_emosi, w_format, w_kred
    low, high = bounds_scale
    lower = np.full(dim, low, dtype=float)
    upper = np.full(dim, high, dtype=float)

    def objective(w: np.ndarray) -> float:
        """
        w[0] = skala emosi
        w[1] = skala format
        w[2] = skala kredibilitas_rendah
        """
        w = np.asarray(w, dtype=float)
        # aman-aman saja kalau PSO masukin nilai di luar, tapi kita clip juga
        w = np.clip(w, lower, upper)

        emo_scaled = np.clip(emo * w[0], 0.0, 1.0)
        frm_scaled = np.clip(frm * w[1], 0.0, 1.0)
        kred_scaled = np.clip(kred * w[2], 0.0, 1.0)

        preds = np.empty_like(y_true)
        for i in range(len(preds)):
            preds[i] = hitung_resiko_hoaks(
                intensitas_emosi=float(emo_scaled[i]),
                kecurigaan_format=float(frm_scaled[i]),
                kredibilitas_rendah=float(kred_scaled[i]),
                output_scale_100=True,
            )

        mse = float(np.mean((preds - y_true) ** 2))
        return mse

    return objective, dim, (lower, upper)


def optimize_fuzzy_scaling_with_pso(
    df: pd.DataFrame,
    target_col: str,
    n_particles: int = 30,
    n_iters: int = 50,
    bounds_scale: Tuple[float, float] = (0.5, 1.5),
    seed: int | None = None,
) -> Dict[str, object]:
    """
    Wrapper utama: jalankan PSO untuk mencari skala input FIS.

    Parameters
    ----------
    df : DataFrame
        Dataset dengan fitur LLM dan label ahli.
    target_col : str
        Nama kolom label (misal 'resiko_hoaks_ahli').
        Nilai biasanya 0–100 (sama skala dengan output fuzzy).
    n_particles, n_iters : int
        Hyperparameter PSO.
    bounds_scale : (float, float)
        Batas bawah & atas untuk skala (default 0.5–1.5).
    seed : int | None
        Seed random.

    Returns
    -------
    dict:
        {
          'best_position': array([w_emosi, w_format, w_kred]),
          'best_fitness': MSE_terbaik,
          'history': list_MSE_per_iter
        }
    """
    objective, dim, bounds = _build_fuzzy_scaling_objective(
        df=df,
        target_col=target_col,
        bounds_scale=bounds_scale,
    )

    result = pso_optimize(
        objective_fn=objective,
        dim=dim,
        bounds=bounds,
        n_particles=n_particles,
        n_iters=n_iters,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=seed,
    )
    return result
