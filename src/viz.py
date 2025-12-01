# src/viz.py

"""
Modul visualisasi untuk sistem fuzzy resiko hoaks.

Berisi helper untuk:
- plot surface 3D: resiko_hoaks vs 2 input
- plot slice 1D: resiko_hoaks vs 1 input (2 input lain tetap)
"""

from __future__ import annotations
from typing import Literal 
import numpy as np
import matplotlib.pyplot as plt
from src.fuzzy_system import hitung_resiko_hoaks



def plot_resiko_surface(
    var_x: Literal["intensitas_emosi", "kecurigaan_format"] = "intensitas_emosi",
    var_y: Literal["kecurigaan_format", "kredibilitas_rendah"] = "kecurigaan_format",
    fixed_value: float = 0.5,
    n_points: int = 40,
    output_scale_100: bool = True,
) -> None:
    """
    Plot permukaan 3D resiko_hoaks terhadap dua variabel input.

    var_x, var_y:
        - kombinasi yang disarankan:
          ("intensitas_emosi", "kecurigaan_format")
          ("intensitas_emosi", "kredibilitas_rendah")
          ("kecurigaan_format", "kredibilitas_rendah")

    fixed_value:
        nilai tetap untuk variabel ketiga (0.0–1.0)

    n_points:
        resolusi grid (semakin besar semakin halus, tapi lebih lama)
    """
    x = np.linspace(0.0, 1.0, n_points)
    y = np.linspace(0.0, 1.0, n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    vars_all = ["intensitas_emosi", "kecurigaan_format", "kredibilitas_rendah"]

    # variabel yang tidak dipakai → dianggap tetap (= fixed_value)
    fixed_var = [v for v in vars_all if v not in (var_x, var_y)]
    if len(fixed_var) != 1:
        raise ValueError("Kombinasi var_x dan var_y tidak valid.")
    fixed_var = fixed_var[0]

    for i in range(n_points):
        for j in range(n_points):
            kwargs = {
                var_x: float(X[i, j]),
                var_y: float(Y[i, j]),
                fixed_var: float(fixed_value),
                "output_scale_100": output_scale_100,
            }
            Z[i, j] = hitung_resiko_hoaks(**kwargs)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (untuk projection='3d')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel(var_x.replace("_", " "))
    ax.set_ylabel(var_y.replace("_", " "))
    ax.set_zlabel("resiko_hoaks" + (" (0–100)" if output_scale_100 else ""))

    ax.set_title(
        f"Surface resiko_hoaks\n{fixed_var} tetap = {fixed_value:.2f}"
    )
    plt.tight_layout()
    plt.show()


def plot_resiko_slice(
    var: Literal["intensitas_emosi", "kecurigaan_format", "kredibilitas_rendah"],
    other1: float = 0.5,
    other2: float = 0.5,
    n_points: int = 100,
    output_scale_100: bool = True,
) -> None:
    """
    Plot kurva 1D resiko_hoaks terhadap satu variabel input.

    var:
        variabel yang di-sweep dari 0–1:
        - "intensitas_emosi"
        - "kecurigaan_format"
        - "kredibilitas_rendah"

    other1, other2:
        nilai tetap untuk dua variabel lainnya (0.0–1.0)
    """
    t = np.linspace(0.0, 1.0, n_points)
    z_vals = []

    vars_all = ["intensitas_emosi", "kecurigaan_format", "kredibilitas_rendah"]
    others = [v for v in vars_all if v != var]
    if len(others) != 2:
        raise ValueError("Nama variabel tidak valid.")

    o1_name, o2_name = others

    for val in t:
        kwargs = {
            var: float(val),
            o1_name: float(other1),
            o2_name: float(other2),
            "output_scale_100": output_scale_100,
        }
        z = hitung_resiko_hoaks(**kwargs)
        z_vals.append(z)

    z_vals = np.array(z_vals)

    plt.figure()
    plt.plot(t, z_vals)
    plt.xlabel(var.replace("_", " "))
    plt.ylabel("resiko_hoaks" + (" (0–100)" if output_scale_100 else ""))
    plt.title(
        f"Slice resiko_hoaks vs {var}\n"
        f"{o1_name} = {other1:.2f}, {o2_name} = {other2:.2f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# src/viz.py  (tambahan)


def mf_tri(x, a, b, c):
    """
    Membership function segitiga:
    a <= b <= c
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    # naik
    idx1 = (x >= a) & (x <= b)
    if b != a:
        y[idx1] = (x[idx1] - a) / (b - a)

    # turun
    idx2 = (x >= b) & (x <= c)
    if c != b:
        y[idx2] = (c - x[idx2]) / (c - b)

    return np.clip(y, 0.0, 1.0)


def plot_hoax_risk_mfs(crisp_score: float | None = None):
    """
    Plot membership function output 'resiko_hoaks' (0–100)
    dengan tiga himpunan fuzzy: low, medium, high (segitiga).

    Parameter segitiga sesuaikan dengan yang kamu pakai di fuzzy_system.py.
    Di sini contoh:
      - low    : tri(0, 0, 40)
      - medium : tri(20, 50, 80)
      - high   : tri(60, 100, 100)
    """

    x = np.linspace(0, 100, 500)

    low = mf_tri(x, 0, 0, 40)
    med = mf_tri(x, 20, 50, 80)
    high = mf_tri(x, 60, 100, 100)

    plt.figure(figsize=(7, 4))
    plt.plot(x, low, label="low")
    plt.plot(x, med, label="medium")
    plt.plot(x, high, label="high")

    # shading biar mirip contohmu
    plt.fill_between(x, 0, low, alpha=0.08)
    plt.fill_between(x, 0, med, alpha=0.08)
    plt.fill_between(x, 0, high, alpha=0.08)

    if crisp_score is not None:
        plt.axvline(crisp_score, color="k", linewidth=2)

    plt.xlabel("hoax_risk (0–100)")
    plt.ylabel("Membership")
    plt.title("Output Fuzzy Set: resiko_hoaks")
    plt.legend()
    plt.ylim(-0.02, 1.05)
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
