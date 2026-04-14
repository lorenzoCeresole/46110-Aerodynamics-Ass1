'''
Author: Lorenzo Ceresole
Course: 46705 - Aerodynamics
Assignment 2 - Rectangular wing with linear twist

Code: Rectancular wing with linear twist
'''
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER INPUTS
# ============================================================
AR = 6.0                       # Rectangular wing aspect ratio
ALPHA_G_ROOT_DEG = 2.0         # Geometric angle at root / midpoint
TIP_ANGLES_DEG = [0, 2, 4, 6, 8]
A0_PER_RAD = 2.0 * np.pi       # 2D lift-curve slope
ALPHA_L0_DEG = -2.0            # Approx. zero-lift AoA for NACA 2410
N_TERMS = 60                   # Number of Fourier terms
N_COLLOCATION = 60             # Number of collocation points
N_PLOT = 801                   # Resolution for smooth plots


# ============================================================
# LIFTING-LINE SOLVER
# ============================================================
def solve_lifting_line_general(
    alpha_geom_deg,
    chord_over_b,
    a0_per_rad=2.0 * np.pi,
    alpha_l0_deg=-2.0,
    n_terms=40,
    n_collocation=40,
):
    """
    Solve Glauert lifting-line system for a planar wing with arbitrary
    geometric incidence and chord distribution.

    Parameters
    ----------
    alpha_geom_deg : callable
        Function of x_tilde in [-1, 1], returning local geometric AoA [deg].
    chord_over_b : callable
        Function of x_tilde in [-1, 1], returning local chord / span.
    a0_per_rad : float
        2D lift-curve slope [1/rad].
    alpha_l0_deg : float
        Zero-lift angle [deg].
    n_terms : int
        Number of Fourier sine coefficients A_n.
    n_collocation : int
        Number of collocation points.

    Returns
    -------
    dict with:
        theta, x_tilde, A, alpha_i_deg, Gamma_tilde, cl_local, cdi_local,
        CL, CDi
    """
    # Collocation points over half-wing transformed to full wing with x_tilde = cos(theta)
    theta = np.arange(1, n_collocation + 1) * np.pi / (n_collocation + 1)
    x_tilde = np.cos(theta)

    alpha_geom_rad = np.deg2rad(alpha_geom_deg(x_tilde))
    alpha_l0_rad = np.deg2rad(alpha_l0_deg)
    alpha_eff_rad = alpha_geom_rad - alpha_l0_rad

    c_over_b = chord_over_b(x_tilde)
    mu = a0_per_rad * c_over_b / 4.0

    # Matrix system:
    # sum_n A_n sin(nθ) [sin(θ) + n μ(θ)] = μ(θ) [α_geom - α_L0] sin(θ)
    n = np.arange(1, n_terms + 1)
    M = np.zeros((n_collocation, n_terms))
    rhs = mu * alpha_eff_rad * np.sin(theta)

    for i, th in enumerate(theta):
        M[i, :] = np.sin(n * th) * (np.sin(th) + n * mu[i])

    A = np.linalg.solve(M, rhs)

    # Use a dense grid for smooth output curves
    theta_plot = np.linspace(1e-6, np.pi - 1e-6, N_PLOT)
    x_tilde_plot = np.cos(theta_plot)

    sin_n_theta = np.sin(np.outer(np.arange(1, n_terms + 1), theta_plot))

    # Gamma = 2 b U sum A_n sin(nθ)
    # Gamma_tilde = Gamma / (c_mean U)
    # For rectangular wing: c_mean = c = b / AR
    # => Gamma_tilde = 2 AR sum A_n sin(nθ)
    gamma_sum = (A[:, None] * sin_n_theta).sum(axis=0)
    Gamma_tilde = 2.0 * AR * gamma_sum

    # alpha_i = sum n A_n sin(nθ)/sinθ
    alpha_i_rad = ((np.arange(1, n_terms + 1)[:, None] * A[:, None]) * sin_n_theta).sum(axis=0) / np.sin(theta_plot)
    alpha_i_deg = np.rad2deg(alpha_i_rad)

    # Local section coefficients
    # c_l = 2 Gamma / (U c) = 4 b/c * sum A_n sin(nθ) = 4 AR * sum(...)
    cl_local = 4.0 * AR * gamma_sum

    # Small-angle induced drag coefficient per section
    cdi_local = cl_local * alpha_i_rad

    # Whole-wing coefficients
    CL = np.pi * AR * A[0]
    CDi = np.pi * AR * np.sum((np.arange(1, n_terms + 1)) * A**2)

    return {
        "theta": theta_plot,
        "x_tilde": x_tilde_plot,
        "A": A,
        "Gamma_tilde": Gamma_tilde,
        "alpha_i_deg": alpha_i_deg,
        "cl_local": cl_local,
        "cdi_local": cdi_local,
        "CL": CL,
        "CDi": CDi,
    }


# ============================================================
# GEOMETRY FOR PART 4
# ============================================================
def rectangular_chord_over_b(x_tilde):
    """
    Rectangular wing with AR = b / c  =>  c / b = 1 / AR
    """
    x_tilde = np.asarray(x_tilde)
    return np.ones_like(x_tilde) / AR


def make_linear_twist(alpha_root_deg, alpha_tip_deg):
    """
    alpha_geom(x_tilde) = alpha_root + (alpha_tip - alpha_root)*|x_tilde|
    """
    def alpha_geom(x_tilde):
        x_tilde = np.asarray(x_tilde)
        return alpha_root_deg + (alpha_tip_deg - alpha_root_deg) * np.abs(x_tilde)

    return alpha_geom


# ============================================================
# PLOTTING
# ============================================================
def style_axes(ax, xlabel=r'$\tilde{x}$'):
    ax.grid(True, alpha=0.35)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(xlabel)


def plot_results(results_by_tip):
    # 1) Dimensionless circulation
    fig1, ax1 = plt.subplots(figsize=(8.0, 4.6))
    for alpha_tip, res in results_by_tip.items():
        ax1.plot(res["x_tilde"], res["Gamma_tilde"], linewidth=1.8, label=f'TR = {alpha_tip:.1f}'.replace("TR = ", r'$\alpha_{g,\mathrm{tip}}$ = ') + r'$^\circ$')
    ax1.set_title(r'Exercise 4 - Dimensionless circulation $\tilde{\Gamma}(\tilde{x})$')
    ax1.set_ylabel(r'$\tilde{\Gamma}$')
    style_axes(ax1)
    ax1.legend()
    fig1.tight_layout()

    # 2) Induced angle
    fig2, ax2 = plt.subplots(figsize=(8.0, 4.6))
    for alpha_tip, res in results_by_tip.items():
        ax2.plot(res["x_tilde"], res["alpha_i_deg"], linewidth=1.8, label=fr'$\alpha_{{g,\mathrm{{tip}}}}$ = {alpha_tip:.1f}$^\circ$')
    ax2.set_title(r'Exercise 4 - Local induced angle $\alpha_i(\tilde{x})$')
    ax2.set_ylabel(r'$\alpha_i$ [deg]')
    style_axes(ax2)
    ax2.legend()
    fig2.tight_layout()

    # 3) Local lift coefficient
    fig3, ax3 = plt.subplots(figsize=(8.0, 4.6))
    for alpha_tip, res in results_by_tip.items():
        ax3.plot(res["x_tilde"], res["cl_local"], linewidth=1.8, label=fr'$\alpha_{{g,\mathrm{{tip}}}}$ = {alpha_tip:.1f}$^\circ$')
    ax3.set_title(r'Exercise 4 - Local lift coefficient $c_l(\tilde{x})$')
    ax3.set_ylabel(r'$c_l$')
    style_axes(ax3)
    ax3.legend()
    fig3.tight_layout()

    # 4) Local induced drag coefficient
    fig4, ax4 = plt.subplots(figsize=(8.0, 4.6))
    for alpha_tip, res in results_by_tip.items():
        ax4.plot(res["x_tilde"], res["cdi_local"], linewidth=1.8, label=fr'$\alpha_{{g,\mathrm{{tip}}}}$ = {alpha_tip:.1f}$^\circ$')
    ax4.set_title(r'Exercise 4 - Local induced drag $c_{d,i}(\tilde{x})$')
    ax4.set_ylabel(r'$c_{d,i}$')
    style_axes(ax4)
    ax4.legend()
    fig4.tight_layout()

    return fig1, fig2, fig3, fig4


# ============================================================
# MAIN
# ============================================================
def main():
    results_by_tip = {}

    for alpha_tip in TIP_ANGLES_DEG:
        alpha_geom_fun = make_linear_twist(ALPHA_G_ROOT_DEG, alpha_tip)
        res = solve_lifting_line_general(
            alpha_geom_deg=alpha_geom_fun,
            chord_over_b=rectangular_chord_over_b,
            a0_per_rad=A0_PER_RAD,
            alpha_l0_deg=ALPHA_L0_DEG,
            n_terms=N_TERMS,
            n_collocation=N_COLLOCATION,
        )
        results_by_tip[alpha_tip] = res

    figs = plot_results(results_by_tip)

    print("\nWhole-wing coefficients for each tip angle")
    print("=" * 56)
    print(f"{'alpha_tip [deg]':>16} | {'CL':>10} | {'CDi':>10}")
    print("-" * 56)
    for alpha_tip, res in results_by_tip.items():
        print(f"{alpha_tip:16.1f} | {res['CL']:10.5f} | {res['CDi']:10.5f}")

    plt.show()


if __name__ == "__main__":
    main()
