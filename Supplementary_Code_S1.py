"""
Supplementary Code S1 - Cluchagues-Marcus Unified Model (in silico)
Reproduces every number and panel of Figure 1 and the statistics in Section 3.
Requires: numpy, scipy, matplotlib.   Run:  python Supplementary_Code_S1.py
All stochastic steps use fixed seeds.  Outputs: Figure1.png/.tiff/.pdf, results.json
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
fig.savefig("Figure1.tiff", dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
print(json.dumps(summ, indent=1))


# =====================================================================
# Sensitivity analysis (Section 3.3, Figure 2)
# Global Monte Carlo over plausible parameter ranges; G is re-calibrated in every draw so that
# lambda_ext(80.4) = lambda_unshielded (Eq. 2).  Outputs: Figure2.png/.pdf and sensitivity.json
# =====================================================================
from scipy.stats import spearmanr
NS = 10000
rs = np.random.default_rng(2026)
RANGES = {  # name: (low, high, scale)
    "dG":      (-0.40, -0.02, "lin"),   # eV, binding free energy
    "eps_s":   (2.0, 4.0, "lin"),       # shielded local dielectric constant
    "eps_opt": (1.4, 1.9, "lin"),       # optical dielectric constant
    "lam_u":   (0.30, 0.70, "lin"),     # eV, unshielded reorganisation energy (calibration)
    "nu":      (1e12, 1e13, "log"),     # s^-1, attempt frequency
    "koff":    (1e10, 1e14, "log"),     # s^-1, k_off,q
    "n":       (0.8, 2.0, "lin"),       # Hill coefficient
}
S = {}
for k, (lo, hi, sc) in RANGES.items():
    S[k] = 10 ** rs.uniform(np.log10(lo), np.log10(hi), NS) if sc == "log" else rs.uniform(lo, hi, NS)
Gs = S["lam_u"] / (1 / S["eps_opt"] - 1 / EPS_U)
lam_s = Gs * (1 / S["eps_opt"] - 1 / S["eps_s"])
kcu = S["nu"] * np.exp(-(S["dG"] + S["lam_u"]) ** 2 / (4 * S["lam_u"] * kT))
kcs = S["nu"] * np.exp(-(S["dG"] + lam_s) ** 2 / (4 * lam_s * kT))
log_ratio = np.log10(kcs / kcu)
ec50_fold = ((kcs + S["koff"]) / (kcu + S["koff"])) ** (1 / S["n"])       # EC50(unshielded)/EC50(shielded)
lam_red = 100 * (1 - lam_s / S["lam_u"])
sens = dict(n_samples=NS, ranges={k: [v[0], v[1], v[2]] for k, v in RANGES.items()},
            lambda_reduction_pct=dict(median=float(np.median(lam_red)), p5=float(np.percentile(lam_red, 5)), p95=float(np.percentile(lam_red, 95))),
            kC_ratio=dict(median=float(np.median(10 ** log_ratio)), p5=float(np.percentile(10 ** log_ratio, 5)), p95=float(np.percentile(10 ** log_ratio, 95))),
            frac_kC_accelerated=float(np.mean(log_ratio > 0)),
            frac_ec50_shifted_lower=float(np.mean(ec50_fold > 1)),
            frac_ec50_fold_above_2=float(np.mean(ec50_fold > 2)),
            ec50_fold=dict(median=float(np.median(ec50_fold)), p5=float(np.percentile(ec50_fold, 5)), p95=float(np.percentile(ec50_fold, 95))))
sub = S["dG"] > -0.15
sens["frac_kC_accelerated_if_dG_gt_minus0p15"] = float(np.mean(log_ratio[sub] > 0))
sens["frac_kC_accelerated_if_dG_le_minus0p15"] = float(np.mean(log_ratio[~sub] > 0))
names = {"dG": r"$\Delta G^{\circ}$", "eps_s": r"$\varepsilon_{\mathrm{shielded}}$", "eps_opt": r"$\varepsilon_{\mathrm{opt}}$",
         "lam_u": r"$\lambda_{\mathrm{ext}}$ (unshielded)", "nu": r"$\nu_{\mathrm{eff}}$", "koff": r"$k_{\mathrm{off,q}}$", "n": r"$n$"}
rho_k = {k: float(spearmanr(S[k], log_ratio)[0]) for k in RANGES}
rho_e = {k: float(spearmanr(S[k], np.log10(ec50_fold))[0]) for k in RANGES}
sens["spearman_log10_kC_ratio"] = rho_k; sens["spearman_log10_EC50_fold"] = rho_e
json.dump(sens, open("sensitivity.json", "w"), indent=1)

fig2 = plt.figure(figsize=(6.5, 5.6))
g2 = fig2.add_gridspec(2, 2, hspace=0.6, wspace=0.55)
bA, bB, bC = fig2.add_subplot(g2[0, 0]), fig2.add_subplot(g2[0, 1]), fig2.add_subplot(g2[1, :])
# A: heat map of log10 rate ratio at default lam_u, eps_opt
dgg = np.linspace(-0.40, -0.02, 200); eps_g = np.linspace(2.0, 4.0, 200)
DGG, EPG = np.meshgrid(dgg, eps_g)
Gd = LAM_U / (1 / EPS_OPT - 1 / EPS_U); lam_g = Gd * (1 / EPS_OPT - 1 / EPG)
Z = np.log10(np.exp(-(DGG + lam_g) ** 2 / (4 * lam_g * kT)) / np.exp(-(DGG + LAM_U) ** 2 / (4 * LAM_U * kT)))
zmax = 4
im = bA.pcolormesh(DGG, EPG, np.clip(Z, -zmax, zmax), cmap="RdBu_r", vmin=-zmax, vmax=zmax, shading="auto", rasterized=True)
bA.contour(DGG, EPG, Z, levels=[0], colors="k", linewidths=1.0)
cb = fig2.colorbar(im, ax=bA, pad=0.03); cb.set_label(r"log$_{10}$ ($k_{\mathrm{C}}$ ratio), clipped at $\pm$4", fontsize=8); cb.ax.tick_params(labelsize=8)
bA.set_xlabel(r"$\Delta G^{\circ}$ (eV)"); bA.set_ylabel(r"$\varepsilon_{\mathrm{shielded}}$")
bA.plot([DG], [EPS_S], "o", color="k", ms=5, mfc="w")
# B: histogram
bins = np.linspace(-8, 3, 45)
bB.hist(np.clip(log_ratio, -8, 3), bins=bins, color=C_S, edgecolor="k", linewidth=.4)
bB.axvline(0, color="k", lw=1, ls="--")
bB.set_xlabel(r"log$_{10}$ ($k_{\mathrm{C}}$ shielded / unshielded)"); bB.set_ylabel("Number of draws")
bB.text(.03, .93, "accelerated: %.0f%%" % (100 * sens["frac_kC_accelerated"]), transform=bB.transAxes, ha="left", va="top", fontsize=8)
# C: Spearman bars
ks = sorted(RANGES, key=lambda k: abs(rho_k[k]) , reverse=True)
yy = np.arange(len(ks)); h = 0.38
bC.barh(yy - h / 2, [rho_k[k] for k in ks], h, color=C_S, edgecolor="k", lw=.5, label=r"$k_{\mathrm{C}}$ ratio")
bC.barh(yy + h / 2, [rho_e[k] for k in ks], h, color=C_U, edgecolor="k", lw=.5, hatch="////", label=r"EC$_{50}$ fold-shift")
bC.set_yticks(yy); bC.set_yticklabels([names[k] for k in ks]); bC.invert_yaxis(); bC.axvline(0, color="k", lw=.8)
bC.set_xlabel("Spearman rank correlation with log$_{10}$ output"); bC.legend(frameon=False, loc="lower right")
for a_, lab in zip([bA, bB, bC], "ABC"): a_.text(-.24 if a_ is not bC else -.14, 1.06, lab, transform=a_.transAxes, fontsize=11, fontweight="bold")
fig2.savefig("Figure2.png", bbox_inches="tight", facecolor="white"); fig2.savefig("Figure2.pdf", bbox_inches="tight", facecolor="white")
fig2.savefig("Figure2.tiff", dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
print(json.dumps(sens, indent=1))


# =====================================================================
# Numerical checks of the analytical propositions (Section 3.4)
#  P1: shielding increases k_C  <=>  |dG| < sqrt(lambda_s * lambda_u)
#  P2: Eq. 3 depends on (Kd, kC, koff) only through EC50  =>  Jacobian rank 4; null dimension 2 / 1 / 0
#  P3: kC <= nu_eff, with equality iff dG = -lambda
# =====================================================================
props = {}
# P1: analytical threshold vs numerical sign change of log(kC_s/kC_u), several (lambda_u, lambda_s) pairs
rp = np.random.default_rng(7)
errs = []
for _ in range(200):
    lu = rp.uniform(0.3, 0.7); ls = lu * rp.uniform(0.05, 0.9)
    grid = -np.linspace(1e-4, 1.0, 200001)
    r = np.log(k_c(ls, grid)) - np.log(k_c(lu, grid))
    idx = np.where(r > 0)[0]
    num = -grid[idx[-1]] if idx.size else 0.0
    errs.append(abs(num - np.sqrt(lu * ls)))
props["P1_threshold_default_eV"] = float(np.sqrt(LAM_U * LAM_S))
props["P1_max_abs_error_vs_numerical_eV"] = float(np.max(errs))
# P2: generic Jacobian rank of the noise-free model for 6, 5 and 4 free parameters
def model_eval(p6, xs=X):
    top, bot, lkd, n, lkc, lko = p6
    return hill(xs, top, bot, 10 ** lkd, n, 10 ** lkc, 10 ** lko)
def num_rank(free, p6):
    J = np.empty((X.size, len(free)))
    for j, i in enumerate(free):
        h = 1e-6 * max(1.0, abs(p6[i])); a = p6.copy(); b = p6.copy(); a[i] += h; b[i] -= h
        J[:, j] = (model_eval(a) - model_eval(b)) / (2 * h)
    sv = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(sv > 1e-6 * sv[0]))
ranks = {"free6": [], "nu5": [], "nu_kd4": []}
for _ in range(500):
    p6 = np.array([rp.uniform(0.8, 1.2), rp.uniform(-0.1, 0.1), rp.uniform(-7, -5), rp.uniform(0.8, 2.0), rp.uniform(10.5, 12.5), rp.uniform(10.5, 12.5)])
    ranks["free6"].append(num_rank([0, 1, 2, 3, 4, 5], p6)); ranks["nu5"].append(num_rank([0, 1, 2, 3, 5], p6)); ranks["nu_kd4"].append(num_rank([0, 1, 3, 5], p6))
props["P2_rank_free6_unique_values"] = sorted(set(ranks["free6"])); props["P2_rank_nu5_unique_values"] = sorted(set(ranks["nu5"])); props["P2_rank_nu_kd4_unique_values"] = sorted(set(ranks["nu_kd4"]))
# P3: bound kC <= nu on a grid, equality at dG = -lambda
gg = np.linspace(-1.0, 0.5, 3001); ll = np.linspace(0.02, 1.0, 400)
GG, LL = np.meshgrid(gg, ll)
kk = NU * np.exp(-(GG + LL) ** 2 / (4 * LL * kT))
props["P3_max_kC_over_nu"] = float(kk.max() / NU)
props["P3_kC_over_nu_at_dG_eq_minus_lambda"] = float(k_c(0.3, -0.3) / NU)
res_all = json.load(open("results.json")); res_all["propositions"] = props
json.dump(res_all, open("results.json", "w"), indent=1)
print(json.dumps(props, indent=1))


# =====================================================================
# Case studies: two deep-pocket targets (Section 3.5, Table 4)
# Reference values (same ligand as in the structure): pocket radii = median ligand-to-lining-atom distance measured with pocket_radius.py (4.5 A cutoff, chain A);
#   KRAS G12C + adagrasib (PDB 6UT0): R_pocket = 0.72 nm, K_I = 1.43 uM (mass-spectrometry assay, FDA review NDA 216340; covalent K_I, not a reversible K_d)
#   STAT3 SH2 + SD-36 (PDB 6NJS): R_pocket = 0.83 nm, K_i = 11 nM (fluorescence-polarization assay, Bai et al. 2019 Cancer Cell)
# The geometric prefactor of Eq. 2 is scaled as G ~ 1/R_pocket (single-cavity Born scaling), calibrated at the KRAS
# reference pocket (G = 0.847 eV  <=>  lambda_ext = 0.46 eV at eps_local = 80.4).
# =====================================================================
R_REF = 0.72   # nm, measured median ligand-to-lining-atom distance (pocket_radius.py) of the KRAS reference pocket (6UT0)
CASES = {"KRAS G12C (6UT0)": dict(R=0.72, Kd=1.43e-6), "STAT3 SH2 (6NJS)": dict(R=0.83, Kd=11e-9)}
case_out = {}
for name, c in CASES.items():
    Gc = G * R_REF / c["R"]
    lu = Gc * (1 / EPS_OPT - 1 / EPS_U); ls = Gc * (1 / EPS_OPT - 1 / EPS_S)
    kcu_c, kcs_c = k_c(lu), k_c(ls)
    ec_u = c["Kd"] * (KOFF / (kcu_c + KOFF)) ** (1 / N_HILL); ec_s = c["Kd"] * (KOFF / (kcs_c + KOFF)) ** (1 / N_HILL)
    TRUEc = np.array([TOP, BOT, np.log10(c["Kd"]), N_HILL, np.log10(kcs_c), np.log10(KOFF)])
    P0c = np.array([1.0, 0.0, np.log10(2 * c["Kd"]), 1.0, np.log10(2e12), np.log10(3e11)])
    def fit_case(cfg, Y):
        free = CFG[cfg]["free"]
        def f(x, *q):
            p = TRUEc.copy(); p[free] = q
            return hill(x, p[0], p[1], 10 ** p[2], p[3], 10 ** p[4], 10 ** p[5])
        popt, pcov, info, *_ = curve_fit(f, X, Y, p0=P0c[free], bounds=(LO[free], HI[free]), method="trf", x_scale="jac", full_output=True, maxfev=400)
        res = Y - f(X, *popt)
        J = np.empty((X.size, len(free)))
        for j in range(len(free)):
            h = 1e-6 * max(1.0, abs(popt[j])); a = popt.copy(); b = popt.copy(); a[j] += h; b[j] -= h
            J[:, j] = (f(X, *a) - f(X, *b)) / (2 * h)
        sv = np.linalg.svd(J, compute_uv=False)
        return dict(r2=1 - np.sum(res ** 2) / np.sum((Y - Y.mean()) ** 2), nfev=int(info["nfev"]), null=len(free) - numerical_rank(sv), popt=popt)
    agg = {k: dict(r2=[], nfev=[], null=[]) for k in CFG}; koffs = []
    for s in range(NSEEDS):
        Yc = hill(X, TOP, BOT, c["Kd"], N_HILL, kcs_c, KOFF) + np.random.default_rng(s).normal(0, SIGMA, NPTS)
        for k in CFG:
            r = fit_case(k, Yc); agg[k]["r2"].append(r["r2"]); agg[k]["nfev"].append(r["nfev"]); agg[k]["null"].append(r["null"])
            if k == "nu_kd": koffs.append(r["popt"][3])
    case_out[name] = dict(R_nm=c["R"], Kd_M=c["Kd"], G_eV=float(Gc), lam_unshielded=float(lu), lam_shielded=float(ls),
        lam_reduction_pct=float(100 * (1 - ls / lu)), kC_unshielded=float(kcu_c), kC_shielded=float(kcs_c), kC_ratio=float(kcs_c / kcu_c),
        EC50_unshielded_M=float(ec_u), EC50_shielded_M=float(ec_s), EC50_fold=float(ec_u / ec_s),
        P1_threshold_eV=float(np.sqrt(lu * ls)),
        identifiability={k: dict(null_dim_median=float(np.median(v["null"])), r2_median=float(np.median(v["r2"])), nfev_median=float(np.median(v["nfev"]))) for k, v in agg.items()},
        koff_log10_mean=float(np.mean(koffs)), koff_log10_sd=float(np.std(koffs, ddof=1)))
res_all = json.load(open("results.json")); res_all["case_studies"] = case_out
json.dump(res_all, open("results.json", "w"), indent=1)
print(json.dumps(case_out, indent=1))


# =====================================================================
# Real data demonstration (Section 3.6, Figure 3, Table 5)
# Input: dose-data_napabucasin.csv  (napabucasin cell-viability curves, 8 concentrations, ~700 cancer cell lines;
# format of the DepMap/PRISM portal download - cite the exact source and version).  Skipped if the file is absent.
# Response = 1 - viability (inhibition fraction).  Fits: four-parameter logistic (4PL) and Eq. 3 with anchors.
# =====================================================================
import os
import pandas as pd
RD = "dose-data_napabucasin.csv"
if os.path.exists(RD):
    df = pd.read_csv(RD, encoding="utf-8-sig")
    dcols = [c for c in df.columns if c.strip().endswith(" uM")]
    conc = np.array([float(c.split()[0]) for c in dcols]) * 1e-6           # M
    def f4(x, top, bot, lec, n):
        return bot + (top - bot) / (1 + (10 ** lec / x) ** n)
    ANCH = [3e-7, 1e-6, 3e-6, 1e-5]                                          # assumed K_d anchors (M), all > typical EC50
    rows = []
    for _, r in df.iterrows():
        v = pd.to_numeric(r[dcols], errors="coerce").values.astype(float)
        if np.sum(np.isfinite(v)) < 7: continue
        m = np.isfinite(v); xs = conc[m]; ys = 1 - v[m]
        try:
            p4, _ = curve_fit(f4, xs, ys, p0=[1, 0, -7, 1], bounds=([0.3, -0.5, -9.5, 0.2], [1.5, 0.6, -4, 5]), maxfev=2000)
        except Exception:
            continue
        res4 = ys - f4(xs, *p4); sse4 = float(np.sum(res4 ** 2)); sst = float(np.sum((ys - ys.mean()) ** 2))
        rec = dict(cell=r["cellLine"], ec50_fit_uM=10 ** p4[2] * 1e6, ec50_data_uM=float(r["ec50"]), r2=1 - sse4 / sst, sse4=sse4, n=p4[3], top=p4[0], bot=p4[1], npts=int(m.sum()))
        # anchored fits of Eq. 3 (k_C fixed, K_d fixed): free = Top, Bottom, n, log10 koff
        for kd in ANCH:
            rr = (10 ** p4[2] / kd) ** p4[3]
            if rr >= 0.999: continue
            ko0 = np.log10(KC_S * rr / (1 - rr))
            g = lambda x, top, bot, nn, lko: hill(x, top, bot, kd, nn, KC_S, 10 ** lko)
            try:
                pa, _ = curve_fit(g, xs, ys, p0=[p4[0], p4[1], p4[3], ko0], bounds=([0.3, -0.5, 0.2, 3], [1.5, 0.6, 5, 16]), maxfev=5000)
                ssea = float(np.sum((ys - g(xs, *pa)) ** 2))
                rec[f"sse_ratio_{kd:g}"] = ssea / sse4; rec[f"log10koff_{kd:g}"] = pa[3]
            except Exception:
                pass
        # Jacobian null dimension of the full model at a parameter vector matching the 4PL optimum (K_d = 1 uM, k_C = KC_S)
        rr = (10 ** p4[2] / 1e-6) ** p4[3]
        if rr < 0.999:
            p6 = np.array([p4[0], p4[1], -6.0, p4[3], np.log10(KC_S), np.log10(KC_S * rr / (1 - rr))])
            def nr(free):
                J = np.empty((xs.size, len(free)))
                for j, i in enumerate(free):
                    h = 1e-6; a = p6.copy(); b = p6.copy(); a[i] += h; b[i] -= h
                    J[:, j] = (hill(xs, a[0], a[1], 10 ** a[2], a[3], 10 ** a[4], 10 ** a[5]) - hill(xs, b[0], b[1], 10 ** b[2], b[3], 10 ** b[4], 10 ** b[5])) / (2 * h)
                sv = np.linalg.svd(J, compute_uv=False); return len(free) - int(np.sum(sv > 1e-6 * sv[0]))
            rec["null_free"] = nr([0, 1, 2, 3, 4, 5]); rec["null_nu"] = nr([0, 1, 2, 3, 5]); rec["null_nu_kd"] = nr([0, 1, 3, 5])
        rows.append(rec)
    R = pd.DataFrame(rows)
    ok = R[R["r2"].notna()]
    lg = np.log10(ok["ec50_fit_uM"] / ok["ec50_data_uM"].clip(lower=1e-6))
    real = dict(n_cell_lines=int(len(ok)), concentrations_uM=[float(c * 1e6) for c in conc],
                r2_median=float(ok["r2"].median()), r2_q25=float(ok["r2"].quantile(.25)), r2_q75=float(ok["r2"].quantile(.75)),
                ec50_fit_median_uM=float(ok["ec50_fit_uM"].median()),
                ec50_agreement_median_abs_log10=float(np.median(np.abs(lg))), ec50_spearman=float(spearmanr(ok["ec50_fit_uM"], ok["ec50_data_uM"])[0]),
                null_dim_free_unique=sorted(set(int(x) for x in R["null_free"].dropna())), null_dim_nu_unique=sorted(set(int(x) for x in R["null_nu"].dropna())),
                null_dim_nu_kd_unique=sorted(set(int(x) for x in R["null_nu_kd"].dropna())))
    for kd in ANCH:
        c = f"sse_ratio_{kd:g}"
        if c in R:
            real[f"anchor_{kd:g}M"] = dict(n=int(R[c].notna().sum()), sse_ratio_median=float(R[c].median()), sse_ratio_p95=float(R[c].quantile(.95)),
                                          log10koff_median=float(R[f"log10koff_{kd:g}"].median()))
    res_all = json.load(open("results.json")); res_all["real_data_napabucasin"] = real
    json.dump(res_all, open("results.json", "w"), indent=1)
    print(json.dumps(real, indent=1))
    # ---- Figure 3
    fig3 = plt.figure(figsize=(6.5, 5.6)); g3 = fig3.add_gridspec(2, 2, hspace=0.6, wspace=0.55)
    cA, cB, cC = fig3.add_subplot(g3[0, :]), fig3.add_subplot(g3[1, 0]), fig3.add_subplot(g3[1, 1])
    srt = ok.sort_values("ec50_fit_uM"); picks = [srt.iloc[int(len(srt) * q)] for q in (0.05, 0.5, 0.95)]
    xx3 = np.logspace(np.log10(conc.min() / 2), np.log10(conc.max() * 2), 300)
    for pk, col, mk in zip(picks, [C_S, C_K, C_U], ["o", "s", "^"]):
        rw = df[df["cellLine"] == pk["cell"]].iloc[0]; v = pd.to_numeric(rw[dcols], errors="coerce").values.astype(float); m = np.isfinite(v)
        cA.semilogx(conc[m] * 1e6, 1 - v[m], mk, color=col, ms=4.5, mfc="none" if mk != "s" else col, label=pk["cell"])
        cA.semilogx(xx3 * 1e6, f4(xx3, pk["top"], pk["bot"], np.log10(pk["ec50_fit_uM"] * 1e-6), pk["n"]), "-", color=col, lw=1.2)
    cA.set_xlabel("Napabucasin (µM)"); cA.set_ylabel("Inhibition (1 − viability)"); cA.legend(frameon=False, loc="upper left", fontsize=7)
    cB.loglog(ok["ec50_data_uM"].clip(lower=1e-4), ok["ec50_fit_uM"], "o", ms=2.5, color=C_S, alpha=.5, mec="none")
    lims = [1e-3, 20]; cB.plot(lims, lims, "k--", lw=.8); cB.set_xlim(lims); cB.set_ylim(lims)
    cB.set_xlabel("EC$_{50}$ in data set (µM)"); cB.set_ylabel("EC$_{50}$ from 4PL fit (µM)")
    kds = [k for k in ANCH if f"anchor_{k:g}M" in real]
    cC.axhline(0, color="k", lw=.8, ls=":")
    pos = np.arange(len(kds)); cC.plot(pos, [100 * (real[f"anchor_{k:g}M"]["sse_ratio_median"] - 1) for k in kds], "s-", color=C_S, ms=5, lw=1.2)
    cC.set_xticks(pos); cC.set_xticklabels(["%g µM" % (k * 1e6) for k in kds]); cC.set_xlabel(r"Anchored $K_{\mathrm{d}}$")
    cC.set_ylabel("Median SSE change vs 4PL (%)", color=C_S); cC.set_ylim(-1, 1); cC.set_xlim(-0.3, len(kds) - 0.7)
    cC.text(0.30, 0.20, "SSE unchanged\n(< 10$^{-7}$ %)", transform=cC.transAxes, ha="center", fontsize=7, color=C_S)
    cC2 = cC.twinx(); cC2.spines["right"].set_visible(True)
    cC2.plot(pos, [real[f"anchor_{k:g}M"]["log10koff_median"] for k in kds], "o-", color=C_U, ms=5); cC2.set_ylabel(r"Median log$_{10}$ $k_{\mathrm{off,q}}$ (s$^{-1}$)", color=C_U)
    for a_, lab, dx in [(cA, "A", -.08), (cB, "B", -.3), (cC, "C", -.3)]: a_.text(dx, 1.06, lab, transform=a_.transAxes, fontsize=11, fontweight="bold")
    fig3.savefig("Figure3.png", bbox_inches="tight", facecolor="white"); fig3.savefig("Figure3.pdf", bbox_inches="tight", facecolor="white")
    try:
        fig3.savefig("Figure3.tiff", dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    except Exception as _e:
        print("TIFF not saved:", _e)
else:
    print("Real-data file not found; real-data section skipped.")


# ---------------- show the figures when run inside a notebook (Colab / Jupyter) ----------------
try:
    from IPython.display import Image as _Img, display as _disp
    for _f in ("Figure1.png", "Figure2.png", "Figure3.png"):
        if os.path.exists(_f):
            print(_f); _disp(_Img(filename=_f))
except Exception:
    pass
print("Done. Output files: Figure1-3 (.png/.pdf/.tiff), results.json, sensitivity.json (see the Files panel on the left in Colab).")

 

