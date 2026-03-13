"""
main.py
Assignment 1 orchestrator

This file:
1) plots airfoil geometries
2) runs thin airfoil theory
3) runs panel method
4) runs XFOIL
5) makes all assignment plots
6) prints the XFOIL max Cl/Cd table

IMPORTANT:
- thin_airfoil_run.py should only expose thin_airfoil(...)
- Xfoil_run.py should only expose run_case(...), read_polar_file(...), read_cp_file(...)
- do NOT let those files auto-plot on import
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from airfoil_toolbox import shape_naca, parse_naca, solve_panel_method
from thin_airfoil_run import thin_airfoil
from Xfoil_run import run_case, read_polar_file, read_cp_file, results_folder


# ============================================================
# USER SETTINGS
# ============================================================

AIRFOILS = [2312, 2324, 4412, 4424]
XFOIL_CASES = ["free", "fixed"]

ALPHA_START = -10
ALPHA_END = 15
ALPHA_STEP = 1
ALPHAS = np.arange(ALPHA_START, ALPHA_END + ALPHA_STEP, ALPHA_STEP)

ALPHA_CP = 10
N_PANELS = 201
N_GEOM = 300

SAVE_FIGURES = True
SHOW_FIGURES = True

FIG_DIR = "Figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# SMALL UTILITIES
# ============================================================

def savefig(name):
    if SAVE_FIGURES:
        path = os.path.join(FIG_DIR, name)
        plt.savefig(path, dpi=300, bbox_inches="tight")


def split_upper_lower_by_y(x, y, cp):
    """
    Split XFOIL surface data into upper and lower surfaces using y sign.

    Assumes:
    - upper surface has y >= 0
    - lower surface has y < 0
    Returns data sorted from LE to TE (in increasing x)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    cp = np.asarray(cp)

    upper_mask = y >= 0
    lower_mask = y < 0

    x_upper = x[upper_mask]
    cp_upper = cp[upper_mask]

    x_lower = x[lower_mask]
    cp_lower = cp[lower_mask]

    # sort both from LE to TE
    iu = np.argsort(x_upper)
    il = np.argsort(x_lower)

    x_upper = x_upper[iu]
    cp_upper = cp_upper[iu]

    x_lower = x_lower[il]
    cp_lower = cp_lower[il]

    return x_upper, cp_upper, x_lower, cp_lower


def panel_upper_lower_from_results(panel_results):
    """
    Split panel-method Cp into upper/lower surfaces using y-coordinate.
    Then sort each from LE to TE.
    """
    xp = np.asarray(panel_results["xp"])
    yp = np.asarray(panel_results["yp"])
    cp = np.asarray(panel_results["Cp"])

    upper_mask = yp >= 0
    lower_mask = yp < 0

    x_upper = xp[upper_mask]
    cp_upper = cp[upper_mask]

    x_lower = xp[lower_mask]
    cp_lower = cp[lower_mask]

    iu = np.argsort(x_upper)
    il = np.argsort(x_lower)

    x_upper = x_upper[iu]
    cp_upper = cp_upper[iu]

    x_lower = x_lower[il]
    cp_lower = cp_lower[il]

    return x_upper, cp_upper, x_lower, cp_lower


def delta_cp_from_surfaces(x_upper, cp_upper, x_lower, cp_lower, x_common=None):
    """
    Computes Delta Cp = Cp_lower - Cp_upper on a common x-grid.
    """
    if x_common is None:
        x_min = max(np.min(x_upper), np.min(x_lower))
        x_max = min(np.max(x_upper), np.max(x_lower))
        x_common = np.linspace(x_min, x_max, 300)

    cp_u_i = np.interp(x_common, x_upper, cp_upper)
    cp_l_i = np.interp(x_common, x_lower, cp_lower)
    delta_cp = cp_l_i - cp_u_i

    return x_common, delta_cp, cp_u_i, cp_l_i


def get_label_from_code(code):
    return f"NACA {code}"


# ============================================================
# GEOMETRY PLOTS
# ============================================================

def plot_geometry_pair(code1, code2, title, filename):
    fig, ax = plt.subplots(figsize=(8, 3))

    for code, color in zip([code1, code2], ["tab:red", "tab:blue"]):
        m, p, xx = parse_naca(code)
        camber, upper, lower = shape_naca(m, p, xx, c=1, N=N_GEOM)

        ax.plot(upper[:, 0], upper[:, 1], color=color, label=get_label_from_code(code))
        ax.plot(lower[:, 0], lower[:, 1], color=color)
        ax.plot(camber[:, 0], camber[:, 1], color="black", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    savefig(filename)


# ============================================================
# DATA GENERATION
# ============================================================

def run_thin_airfoil_for_code(code):
    m, p, xx = parse_naca(code)

    cl_list = []
    for alpha in ALPHAS:
        result = thin_airfoil(m, p, alpha)
        cl_list.append(result["Cl"])

    cp10 = thin_airfoil(m, p, ALPHA_CP)

    return {
        "alpha": ALPHAS.copy(),
        "cl": np.asarray(cl_list),
        "x_delta": np.asarray(cp10["x/c"]),
        "delta_cp": np.asarray(cp10["delta_Cp"]),
    }


def run_panel_for_code(code):
    cl_list = []
    for alpha in ALPHAS:
        result = solve_panel_method(code, alpha, N=N_PANELS)
        cl_list.append(result["Cl"])

    cp10 = solve_panel_method(code, ALPHA_CP, N=N_PANELS)

    x_upper, cp_upper, x_lower, cp_lower = panel_upper_lower_from_results(cp10)
    x_delta, delta_cp, _, _ = delta_cp_from_surfaces(x_upper, cp_upper, x_lower, cp_lower)

    return {
        "alpha": ALPHAS.copy(),
        "cl": np.asarray(cl_list),
        "cp10_raw": cp10,
        "x_delta": x_delta,
        "delta_cp": delta_cp,
    }


def run_xfoil_for_code(code):
    code_str = str(code)
    out = {}

    for case in XFOIL_CASES:
        polar_path, cp_path = run_case(code_str, case)

        alpha, cl, cd = read_polar_file(polar_path)

        # IMPORTANT:
        # read_cp_file() should return x, y, cp
        # not just x, cp
        x_cp, y_cp, cp = read_cp_file(cp_path)

        x_upper, cp_upper, x_lower, cp_lower = split_upper_lower_by_y(x_cp, y_cp, cp)
        x_delta, delta_cp, _, _ = delta_cp_from_surfaces(x_upper, cp_upper, x_lower, cp_lower)

        out[case] = {
            "alpha": alpha,
            "cl": cl,
            "cd": cd,
            "x_cp": x_cp,
            "y_cp": y_cp,
            "cp": cp,
            "x_upper": x_upper,
            "cp_upper": cp_upper,
            "x_lower": x_lower,
            "cp_lower": cp_lower,
            "x_delta": x_delta,
            "delta_cp": delta_cp,
        }

    return out


def run_all_methods():
    all_results = {}

    for code in AIRFOILS:
        print(f"Running NACA {code} ...")
        all_results[code] = {
            "thin": run_thin_airfoil_for_code(code),
            "panel": run_panel_for_code(code),
            "xfoil": run_xfoil_for_code(code),
        }

    return all_results


# ============================================================
# PLOTS
# ============================================================

def plot_cl_vs_alpha(code, results):
    fig, ax = plt.subplots(figsize=(7, 4))

    thin = results["thin"]
    panel = results["panel"]
    xfoil = results["xfoil"]

    ax.plot(thin["alpha"], thin["cl"], linewidth=2, label="Thin airfoil")
    ax.plot(panel["alpha"], panel["cl"], linewidth=2, label="Panel method")
    ax.plot(xfoil["free"]["alpha"], xfoil["free"]["cl"], "o-", markersize=4, label="XFOIL free")
    ax.plot(xfoil["fixed"]["alpha"], xfoil["fixed"]["cl"], "s-", markersize=4, label="XFOIL fixed")

    ax.set_title(f"$C_l$ vs AoA — NACA {code}")
    ax.set_xlabel("Angle of attack (deg)")
    ax.set_ylabel("$C_l$")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    savefig(f"NACA_{code}_Cl_vs_AoA.png")


def plot_delta_cp(code, results):
    fig, ax = plt.subplots(figsize=(7, 4))

    thin = results["thin"]
    panel = results["panel"]
    xfoil = results["xfoil"]

    ax.plot(thin["x_delta"], thin["delta_cp"], linewidth=2, label="Thin airfoil")
    ax.plot(panel["x_delta"], panel["delta_cp"], linewidth=2, label="Panel method")
    ax.plot(xfoil["free"]["x_delta"], xfoil["free"]["delta_cp"], linewidth=2, label="XFOIL free")
    ax.plot(xfoil["fixed"]["x_delta"], xfoil["fixed"]["delta_cp"], linewidth=2, label="XFOIL fixed")

    ax.set_title(f"$\\Delta C_p$ vs x/c at AoA = {ALPHA_CP}° — NACA {code}")
    ax.set_xlabel("x/c")
    ax.set_ylabel("$\\Delta C_p$")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    savefig(f"NACA_{code}_DeltaCp_vs_xc.png")


def plot_cp(code, results):
    fig, ax = plt.subplots(figsize=(7, 4))

    panel = results["panel"]["cp10_raw"]
    xfoil = results["xfoil"]

    ax.plot(panel["xp"], panel["Cp"], linewidth=2, label="Panel method")
    ax.plot(xfoil["free"]["x_cp"], xfoil["free"]["cp"], linewidth=2, label="XFOIL free")
    ax.plot(xfoil["fixed"]["x_cp"], xfoil["fixed"]["cp"], linewidth=2, label="XFOIL fixed")

    ax.invert_yaxis()
    ax.set_title(f"$C_p$ vs x/c at AoA = {ALPHA_CP}° — NACA {code}")
    ax.set_xlabel("x/c")
    ax.set_ylabel("$C_p$")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    savefig(f"NACA_{code}_Cp_vs_xc.png")


def plot_xfoil_polar(code, results):
    fig, ax = plt.subplots(figsize=(7, 4))

    xfoil = results["xfoil"]

    for case, marker in zip(XFOIL_CASES, ["o-", "s-"]):
        cd = xfoil[case]["cd"]
        cl = xfoil[case]["cl"]
        ax.plot(cd, cl, marker, markersize=4, label=f"XFOIL {case}")

    ax.set_title(f"XFOIL polar — NACA {code}")
    ax.set_xlabel("$C_d$")
    ax.set_ylabel("$C_l$")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    savefig(f"NACA_{code}_XFOIL_polar.png")


# ============================================================
# TABLE FOR MAX CL/CD
# ============================================================

def build_xfoil_summary_table(all_results):
    rows = []

    for code in AIRFOILS:
        for case in XFOIL_CASES:
            alpha = all_results[code]["xfoil"][case]["alpha"]
            cl = all_results[code]["xfoil"][case]["cl"]
            cd = all_results[code]["xfoil"][case]["cd"]

            valid = cd > 0
            alpha = alpha[valid]
            cl = cl[valid]
            cd = cd[valid]

            if len(alpha) == 0:
                rows.append((code, case, np.nan, np.nan))
                continue

            eff = cl / cd
            i_best = np.argmax(eff)

            rows.append((code, case, eff[i_best], alpha[i_best]))

    return rows


def print_xfoil_summary_table(rows):
    print("\n" + "=" * 72)
    print("XFOIL: MAX Cl/Cd")
    print("=" * 72)
    print(f"{'Case':<28} {'Max Cl/Cd':>14} {'AoA at max (deg)':>18}")
    print("-" * 72)

    for code, case, best_eff, best_alpha in rows:
        name = f"NACA {code} {case}"
        if np.isnan(best_eff):
            print(f"{name:<28} {'FAILED':>14} {'FAILED':>18}")
        else:
            print(f"{name:<28} {best_eff:>14.3f} {best_alpha:>18.1f}")
    print("=" * 72 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    # 1) geometry plots
    plot_geometry_pair(2312, 2324, "NACA 2312 and 2324", "geometry_2312_2324.png")
    plot_geometry_pair(4412, 4424, "NACA 4412 and 4424", "geometry_4412_4424.png")

    # 2) run all methods
    all_results = run_all_methods()

    # 3) per-airfoil plots
    for code in AIRFOILS:
        plot_cl_vs_alpha(code, all_results[code])
        plot_delta_cp(code, all_results[code])
        plot_cp(code, all_results[code])
        plot_xfoil_polar(code, all_results[code])

    # 4) summary table
    rows = build_xfoil_summary_table(all_results)
    print_xfoil_summary_table(rows)

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()