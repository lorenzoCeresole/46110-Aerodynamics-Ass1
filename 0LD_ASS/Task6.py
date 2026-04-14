"""
Author: Lorenzo Ceresole
Class: 46110 Fundamentals of Aerodynamics
-----------------------------------------
Question 6 study with XFOIL:
- Reynolds number sensitivity
- Ncrit sensitivity
- fixed transition location sensitivity
-----------------------------------------
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

# path to xfoil
xfoil_path = r"C:\Users\Lorenzo Ceresole\Documents\Wind Turby\Xfoil\xfoil.exe"

# NACA airfoils
airfoils = ["2312", "2324", "4412", "4424"]

# angle of attack range
alpha_start = -10
alpha_end = 15
alpha_step = 1

# flow settings
mach = 0.0
iterations = 200

# baseline from assignment
re_baseline = 1.5e6
ncrit_baseline = 9
xtr_baseline = 0.1   # upper surface fixed transition location

# studies for question 6
reynolds_list = [1.5e6, 1.5e5]      # baseline + one order of magnitude lower
ncrit_list = [5, 9, 12]             # lower, baseline, higher
xtr_list = [0.05, 0.10, 0.20, 0.40] # earlier and later forced transition

# output folders
results_root = "xfoil_q6_results"
polar_folder = os.path.join(results_root, "polars")
plot_folder = os.path.join(results_root, "plots")

os.makedirs(results_root, exist_ok=True)
os.makedirs(polar_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)


# ============================================================
# XFOIL RUNNER
# ============================================================
def run_xfoil_case(airfoil, re, mode="free", ncrit=9, xtr=0.1):
    """
    mode:
        "free"  -> free transition using Ncrit
        "fixed" -> fixed transition on upper side at xtr, lower side free
    """

    if mode == "free":
        polar_file = f"NACA{airfoil}_free_Re{int(re):d}_N{ncrit}.txt"
    else:
        polar_file = f"NACA{airfoil}_fixed_Re{int(re):d}_XTR{xtr:.2f}.txt"

    polar_path = os.path.join(polar_folder, polar_file)

    # remove old file so XFOIL does not append
    if os.path.exists(polar_path):
        os.remove(polar_path)

    commands = []
    commands.append(f"NACA {airfoil}")
    commands.append("OPER")
    commands.append(f"VISC {re}")
    commands.append(f"MACH {mach}")
    commands.append(f"ITER {iterations}")
    commands.append("VPAR")
    commands.append(f"N {ncrit}")

    if mode == "fixed":
        # upper side forced at xtr, lower side free
        commands.append(f"XTR {xtr} 1")

    commands.append("")
    commands.append("PACC")
    commands.append(polar_file)
    commands.append("")
    commands.append(f"ASEQ {alpha_start} {alpha_end} {alpha_step}")
    commands.append("PACC")
    commands.append("")
    commands.append("QUIT")

    xfoil_input = "\n".join(commands) + "\n"

    try:
        subprocess.run(
            xfoil_path,
            input=xfoil_input,
            text=True,
            cwd=polar_folder,
            check=False
        )
    except Exception as e:
        print(f"XFOIL failed for NACA {airfoil}, mode={mode}, Re={re}: {e}")

    return polar_path


# ============================================================
# POLAR READER
# ============================================================
def read_polar_file(file_path):
    alpha = []
    cl = []
    cd = []
    cm = []

    if not os.path.exists(file_path):
        return np.array([]), np.array([]), np.array([]), np.array([])

    with open(file_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                a = float(parts[0])
                cl_val = float(parts[1])
                cd_val = float(parts[2])
                cm_val = float(parts[4])

                alpha.append(a)
                cl.append(cl_val)
                cd.append(cd_val)
                cm.append(cm_val)
            except:
                pass

    alpha = np.array(alpha)
    cl = np.array(cl)
    cd = np.array(cd)
    cm = np.array(cm)

    if len(alpha) == 0:
        return alpha, cl, cd, cm

    # remove duplicates
    alpha_unique, idx = np.unique(alpha, return_index=True)
    cl = cl[idx]
    cd = cd[idx]
    cm = cm[idx]
    alpha = alpha_unique

    # sort by alpha
    sort_idx = np.argsort(alpha)
    alpha = alpha[sort_idx]
    cl = cl[sort_idx]
    cd = cd[sort_idx]
    cm = cm[sort_idx]

    return alpha, cl, cd, cm


# ============================================================
# HELPERS
# ============================================================
def safe_ld(cl, cd):
    with np.errstate(divide="ignore", invalid="ignore"):
        ld = cl / cd
    ld[~np.isfinite(ld)] = np.nan
    return ld


def save_summary_line(summary_file, airfoil, label, alpha, cl, cd):
    if len(alpha) == 0:
        with open(summary_file, "a") as f:
            f.write(f"{airfoil:8s} | {label:30s} | NO DATA\n")
        return

    ld = safe_ld(cl, cd)
    if np.all(np.isnan(ld)):
        with open(summary_file, "a") as f:
            f.write(f"{airfoil:8s} | {label:30s} | INVALID L/D\n")
        return

    i_best = np.nanargmax(ld)
    with open(summary_file, "a") as f:
        f.write(
            f"{airfoil:8s} | {label:30s} | "
            f"max Cl/Cd = {ld[i_best]:8.3f} | "
            f"AoA = {alpha[i_best]:6.2f} deg | "
            f"Cl = {cl[i_best]:7.4f} | Cd = {cd[i_best]:9.6f}\n"
        )


# ============================================================
# RUN STUDIES
# ============================================================
all_data = {
    "re_study": {},
    "ncrit_study": {},
    "xtr_study": {}
}

summary_file = os.path.join(results_root, "summary.txt")
with open(summary_file, "w") as f:
    f.write("XFOIL QUESTION 6 SUMMARY\n")
    f.write("=" * 90 + "\n\n")

# ----------------------------
# 1) Reynolds number study
# ----------------------------
for airfoil in airfoils:
    all_data["re_study"][airfoil] = {}

    for re in reynolds_list:
        # free transition
        polar_path = run_xfoil_case(
            airfoil=airfoil,
            re=re,
            mode="free",
            ncrit=ncrit_baseline
        )
        alpha, cl, cd, cm = read_polar_file(polar_path)

        label = f"free_Re{int(re):d}_N{ncrit_baseline}"
        all_data["re_study"][airfoil][label] = {
            "alpha": alpha,
            "cl": cl,
            "cd": cd,
            "cm": cm
        }
        save_summary_line(summary_file, airfoil, label, alpha, cl, cd)

        # fixed transition
        polar_path = run_xfoil_case(
            airfoil=airfoil,
            re=re,
            mode="fixed",
            ncrit=ncrit_baseline,
            xtr=xtr_baseline
        )
        alpha, cl, cd, cm = read_polar_file(polar_path)

        label = f"fixed_Re{int(re):d}_XTR{xtr_baseline:.2f}"
        all_data["re_study"][airfoil][label] = {
            "alpha": alpha,
            "cl": cl,
            "cd": cd,
            "cm": cm
        }
        save_summary_line(summary_file, airfoil, label, alpha, cl, cd)

# ----------------------------
# 2) Ncrit study
#    (free transition only)
# ----------------------------
for airfoil in airfoils:
    all_data["ncrit_study"][airfoil] = {}

    for ncrit in ncrit_list:
        polar_path = run_xfoil_case(
            airfoil=airfoil,
            re=re_baseline,
            mode="free",
            ncrit=ncrit
        )
        alpha, cl, cd, cm = read_polar_file(polar_path)

        label = f"free_Re{int(re_baseline):d}_N{ncrit}"
        all_data["ncrit_study"][airfoil][label] = {
            "alpha": alpha,
            "cl": cl,
            "cd": cd,
            "cm": cm
        }
        save_summary_line(summary_file, airfoil, label, alpha, cl, cd)

# ----------------------------
# 3) Fixed transition study
#    (vary upper-surface XTR)
# ----------------------------
for airfoil in airfoils:
    all_data["xtr_study"][airfoil] = {}

    for xtr in xtr_list:
        polar_path = run_xfoil_case(
            airfoil=airfoil,
            re=re_baseline,
            mode="fixed",
            ncrit=ncrit_baseline,
            xtr=xtr
        )
        alpha, cl, cd, cm = read_polar_file(polar_path)

        label = f"fixed_Re{int(re_baseline):d}_XTR{xtr:.2f}"
        all_data["xtr_study"][airfoil][label] = {
            "alpha": alpha,
            "cl": cl,
            "cd": cd,
            "cm": cm
        }
        save_summary_line(summary_file, airfoil, label, alpha, cl, cd)


# ============================================================
# PLOTTING FUNCTION
# ============================================================
def make_plots(study_name, study_data):
    for airfoil in airfoils:
        case_data = study_data[airfoil]

        # ---- Cl vs AoA ----
        plt.figure(figsize=(8, 5))
        for label, data in case_data.items():
            plt.plot(data["alpha"], data["cl"], marker="o", label=label)
        plt.xlabel("AoA (deg)")
        plt.ylabel("$C_l$")
        plt.title(f"{study_name} - NACA {airfoil} - $C_l$ vs AoA")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"{study_name}_NACA{airfoil}_cl.png"), dpi=300)
        plt.close()

        # ---- Cd vs AoA ----
        plt.figure(figsize=(8, 5))
        for label, data in case_data.items():
            plt.plot(data["alpha"], data["cd"], marker="o", label=label)
        plt.xlabel("AoA (deg)")
        plt.ylabel("$C_d$")
        plt.title(f"{study_name} - NACA {airfoil} - $C_d$ vs AoA")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"{study_name}_NACA{airfoil}_cd.png"), dpi=300)
        plt.close()

        # ---- Cl/Cd vs AoA ----
        plt.figure(figsize=(8, 5))
        for label, data in case_data.items():
            ld = safe_ld(data["cl"], data["cd"])
            plt.plot(data["alpha"], ld, marker="o", label=label)
        plt.xlabel("AoA (deg)")
        plt.ylabel("$C_l/C_d$")
        plt.title(f"{study_name} - NACA {airfoil} - $C_l/C_d$ vs AoA")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"{study_name}_NACA{airfoil}_ld.png"), dpi=300)
        plt.close()

        # ---- Polar: Cl vs Cd ----
        plt.figure(figsize=(8, 5))
        for label, data in case_data.items():
            plt.plot(data["cd"], data["cl"], marker="o", label=label)
        plt.xlabel("$C_d$")
        plt.ylabel("$C_l$")
        plt.title(f"{study_name} - NACA {airfoil} - Polar")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"{study_name}_NACA{airfoil}_polar.png"), dpi=300)
        plt.close()


# make all plots
make_plots("Re Change", all_data["re_study"])
make_plots("Ncrit Changey", all_data["ncrit_study"])

print("Done.")
print(f"Polar files saved in: {polar_folder}")
print(f"Plots saved in:      {plot_folder}")
print(f"Summary saved in:    {summary_file}")