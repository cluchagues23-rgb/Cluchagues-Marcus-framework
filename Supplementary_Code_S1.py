"""
Supplementary Code S1 - Cluchagues-Marcus Unified Model (in silico)
Reproduces every number and panel of Figure 1 and the statistics in Section 3.
Requires: numpy, scipy, matplotlib.   Run:  python Supplementary_Code_S1.py
All stochastic steps use fixed seeds.  Outputs: Figure1.png, Figure1.pdf, results.json
"""
import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
# ---------------- constants and parameters ----------------
kB = 8.617333262e-5; T = 310.15; kT = kB * T        # eV
NU = 6.46e12                                         # s^-1, fixed a priori
EPS_U, EPS_S, EPS_OPT = 80.4, 2.0, 1.8
LAM_U = 0.46                                         # eV, unshielded value (calibration point)
# Eq. 2 geometric prefactor G = (De)^2/(4 pi eps0) (1/2rD + 1/2rA - 1/RDA), calibrated so that lambda(80.4) = 0.46 eV
G = LAM_U / (1 / EPS_OPT - 1 / EPS_U)
lam = lambda eps: G * (1 / EPS_OPT - 1 / eps)
LAM_S = lam(EPS_S)
DG = -0.10                                           # eV  (assumed)
KD, KOFF, N_HILL, TOP, BOT = 1.0e-6, 1.0e12, 1.2, 1.0, 0.0   # assumed (illustrative)
SIGMA, NPTS, NSEEDS = 0.05, 100, 200
X = np.logspace(-10, -4, NPTS)

def k_c(l, dg=DG, nu=NU):
    return nu * np.exp(-(dg + l) ** 2 / (4 * l * kT))
def hill(x, top, bot, kd, n, kc, koff):
    return bot + (top - bot) / (1 + (kd * (koff / (kc + koff)) ** (1 / n) / x) ** n)
def ec50(kc):
    return KD * (KOFF / (kc + KOFF)) ** (1 / N_HILL)

KC_U, KC_S = k_c(LAM_U), k_c(LAM_S)

# ---------------- fitting configurations (TRR via curve_fit, bounds => method 'trf') ----------------
# parameters: [Top, Bottom, log10 Kd, n, log10 kC, log10 koff]
CFG = {
 "free":   dict(label=r"ν$_{\rm eff}$ free (6 parameters)",             free=[0,1,2,3,4,5]),
 "nu":     dict(label=r"ν$_{\rm eff}$ fixed (5 parameters)",            free=[0,1,2,3,5]),
 "nu_kd":  dict(label=r"ν$_{\rm eff}$ and $K_{\rm d}$ fixed (4 parameters)", free=[0,1,3,5]),
}
TRUE = np.array([TOP, BOT, np.log10(KD), N_HILL, np.log10(KC_S), np.log10(KOFF)])
P0   = np.array([1.0, 0.0, np.log10(2e-6), 1.0, np.log10(2e12), np.log10(3e11)])
LO   = np.array([0.5, -0.5, -9, 0.3, 8, 6]); HI = np.array([1.5, 0.5, -3, 4, 14, 15])
FIXED = TRUE.copy()                                   # values used for fixed parameters

def wrap(free):
    def f(x, *q):
        p = FIXED.copy(); p[free] = q
        return hill(x, p[0], p[1], 10 ** p[2], p[3], 10 ** p[4], 10 ** p[5])
    return f

def fit(cfg, Y):
    free = CFG[cfg]["free"]; f = wrap(free)
    popt, pcov, info, *_ = curve_fit(f, X, Y, p0=P0[free], bounds=(LO[free], HI[free]),
                                     method="trf", x_scale="jac", full_output=True, maxfev=400)
    res = Y - f(X, *popt)
    # central-difference Jacobian at the optimum, scaled columns
    J = np.empty((X.size, len(free)))
    for j in range(len(free)):
        h = 1e-6 * max(1.0, abs(popt[j])); a = popt.copy(); b = popt.copy(); a[j] += h; b[j] -= h
        J[:, j] = (f(X, *a) - f(X, *b)) / (2 * h)
    sv = np.linalg.svd(J, compute_uv=False)
    r2 = 1 - np.sum(res ** 2) / np.sum((Y - Y.mean()) ** 2)
    return dict(popt=popt, nfev=int(info["nfev"]), r2=float(r2), rmse=float(np.sqrt(np.mean(res ** 2))), sv=sv, free=free)

def numerical_rank(sv, tol=1e-6):
    return int(np.sum(sv > tol * sv[0]))

# ---------------- statistics over noise realisations (shielded condition) ----------------
stats = {k: dict(r2=[], rmse=[], nfev=[], nulldim=[]) for k in CFG}
koff_est = []
for s in range(NSEEDS):
    Y = hill(X, TOP, BOT, KD, N_HILL, KC_S, KOFF) + np.random.default_rng(s).normal(0, SIGMA, NPTS)
    for k in CFG:
        r = fit(k, Y)
        stats[k]["r2"].append(r["r2"]); stats[k]["rmse"].append(r["rmse"]); stats[k]["nfev"].append(r["nfev"])
        stats[k]["nulldim"].append(len(r["free"]) - numerical_rank(r["sv"]))
        if k == "nu_kd": koff_est.append(r["popt"][3])
summ = {}
for k, d in stats.items():
    summ[k] = dict(n_params=len(CFG[k]["free"]),
                   null_dimensions_median=float(np.median(d["nulldim"])),
                   r2_median=float(np.median(d["r2"])), r2_q25=float(np.percentile(d["r2"], 25)), r2_q75=float(np.percentile(d["r2"], 75)),
                   frac_r2_above_0992=float(np.mean(np.array(d["r2"]) > 0.992)),
                   rmse_median=float(np.median(d["rmse"])),
                   nfev_median=float(np.median(d["nfev"])), nfev_q25=float(np.percentile(d["nfev"], 25)), nfev_q75=float(np.percentile(d["nfev"], 75)),
                   frac_nfev_below_15=float(np.mean(np.array(d["nfev"]) < 15)))
koff_est = np.array(koff_est)
summ["koff_recovery_nu_kd"] = dict(true_log10=float(np.log10(KOFF)), mean_log10=float(koff_est.mean()), sd_log10=float(koff_est.std(ddof=1)))

# representative dataset (seed 42) for plots
rng = np.random.default_rng(42)
Yu = hill(X, TOP, BOT, KD, N_HILL, KC_U, KOFF) + rng.normal(0, SIGMA, NPTS)
Ys = hill(X, TOP, BOT, KD, N_HILL, KC_S, KOFF) + rng.normal(0, SIGMA, NPTS)
rep = {k: fit(k, Ys) for k in CFG}

# dG sensitivity of shielded/unshielded rate ratio
dgs = np.linspace(-0.70, 0.0, 281)
ratio = np.array([k_c(LAM_S, d) / k_c(LAM_U, d) for d in dgs])
summ["model"] = dict(G_eV=float(G), lam_unshielded=LAM_U, lam_shielded=float(LAM_S), lam_reduction_pct=float(100 * (1 - LAM_S / LAM_U)),
                     kC_unshielded=float(KC_U), kC_shielded=float(KC_S), kC_ratio=float(KC_S / KC_U),
                     ec50_unshielded=float(ec50(KC_U)), ec50_shielded=float(ec50(KC_S)), ec50_fold=float(ec50(KC_U) / ec50(KC_S)),
                     dG_break_even_range=[float(dgs[ratio > 1][0]), float(dgs[ratio > 1][-1])],
                     max_possible_kC=NU)
json.dump(summ, open("results.json", "w"), indent=1)

# ---------------- Figure 1 (colour-blind-safe Okabe-Ito palette) ----------------
C_U, C_S, C_K = "#D55E00", "#0072B2", "#009E73"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
fig, ax = plt.subplots(2, 2, figsize=(6.5, 6.3))
fig.subplots_adjust(hspace=0.6, wspace=0.6)
axA, axB, axC, axD = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

# A
axA2 = axA.twinx(); axA2.spines["right"].set_visible(True)
axA.bar([0, 1], [EPS_U, EPS_S], .8, color=[C_U, C_S], edgecolor="k", lw=.8); axA.bar([1], [EPS_S], .8, color=C_S, edgecolor="k", hatch="////", lw=.8)
axA2.bar([2.8, 3.8], [LAM_U, LAM_S], .8, color=[C_U, C_S], edgecolor="k", lw=.8); axA2.bar([3.8], [LAM_S], .8, color=C_S, edgecolor="k", hatch="////", lw=.8)
for x, v in [(0, EPS_U), (1, EPS_S)]: axA.text(x, v + 2, f"{v:g}", ha="center", fontsize=8)
for x, v in [(2.8, LAM_U), (3.8, LAM_S)]: axA2.text(x, v + .012, f"{v:.2f}", ha="center", fontsize=8)
axA.set_ylim(0, 100); axA2.set_ylim(0, .6); axA.set_xlim(-.6, 4.4)
axA.set_ylabel(r"$\varepsilon_{\mathrm{local}}$"); axA2.set_ylabel(r"$\lambda_{\mathrm{ext}}$ (eV)")
axA.set_xticks([0, 1, 2.8, 3.8]); axA.set_xticklabels(["Un-\nshielded", "Shielded", "Un-\nshielded", "Shielded"]); axA.tick_params(axis="x", length=0)
axA.text(.5, -.26, r"$\varepsilon_{\mathrm{local}}$ (left)", transform=axA.get_xaxis_transform(), ha="center", va="top")
axA.text(3.3, -.26, r"$\lambda_{\mathrm{ext}}$ (right)", transform=axA.get_xaxis_transform(), ha="center", va="top")

# B: rate ratio vs dG
axB.semilogy(dgs, ratio, color=C_K, lw=1.6)
axB.axhline(1, color="k", lw=.8, ls=":")
axB.axvline(DG, color="k", lw=.8, ls="--")
axB.plot([DG], [KC_S / KC_U], "o", color=C_K, ms=5, mec="k")
axB.text(DG + .02, 1e-7, r"$\Delta G^{\circ}$ = %.2f eV" % DG, fontsize=8, ha="left")
axB.set_xlabel(r"$\Delta G^{\circ}$ (eV)"); axB.set_ylabel(r"$k_{\mathrm{C}}$ (shielded) / $k_{\mathrm{C}}$ (unshielded)")
axB.set_ylim(1e-12, 1e3)
axB.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=100, numticks=8))

# C: dose-response
xx = np.logspace(-10, -4, 400)
axC.semilogx(X, Yu, "o", ms=2.8, mfc="none", mec=C_U, mew=.7, alpha=.9); axC.semilogx(X, Ys, "s", ms=2.8, mfc=C_S, mec=C_S, mew=.4, alpha=.7)
axC.semilogx(xx, hill(xx, TOP, BOT, KD, N_HILL, KC_U, KOFF), "-", color=C_U, lw=1.5, label="Unshielded")
axC.semilogx(xx, hill(xx, TOP, BOT, KD, N_HILL, KC_S, KOFF), "--", color=C_S, lw=1.5, label="Shielded")
for kc, c in [(KC_U, C_U), (KC_S, C_S)]: axC.vlines(ec50(kc), -.12, .5, colors=c, linestyles=":", lw=1)
axC.text(.03, .60, r"EC$_{50}$ shift: %.1f-fold" % (ec50(KC_U) / ec50(KC_S)), transform=axC.transAxes)
axC.set_xlabel("Ligand concentration, X (M)"); axC.set_ylabel("Response, Y (fraction)")
axC.set_ylim(-.15, 1.2); axC.set_xlim(1e-10, 1e-4); axC.legend(frameon=False, loc="upper left", borderaxespad=.2)

# D: singular values
styles = {"free": ("s", C_U, "--"), "nu": ("^", C_S, "-"), "nu_kd": ("o", C_K, "-")}
for k, (m, c, ls) in styles.items():
    sv = rep[k]["sv"] / rep[k]["sv"][0]
    axD.semilogy(np.arange(1, len(sv) + 1), sv, marker=m, color=c, ls=ls, lw=1.2, ms=5, mfc=c if k != "free" else "none",
                 label=f"{CFG[k]['label'].split('(')[0].strip()}: null dim. {len(sv) - numerical_rank(rep[k]['sv'])}")
axD.axhline(1e-6, color="k", lw=.8, ls=":"); axD.text(6.2, 1.6e-6, "threshold", ha="right", va="bottom", fontsize=8)
axD.set_xlabel("Singular value index"); axD.set_ylabel("Normalised singular value of Jacobian")
axD.set_xticks(range(1, 7)); axD.set_ylim(1e-12, 5); axD.set_xlim(0.7, 6.3); axD.legend(frameon=False, loc="center left", fontsize=7, bbox_to_anchor=(-.02, .36), handlelength=1.6)
for a, lab in zip([axA, axB, axC, axD], "ABCD"):
    a.text(-.30, 1.06, lab, transform=a.transAxes, fontsize=11, fontweight="bold")
fig.savefig("Figure1.png", bbox_inches="tight", facecolor="white"); fig.savefig("Figure1.pdf", bbox_inches="tight", facecolor="white")
print(json.dumps(summ, indent=1))
