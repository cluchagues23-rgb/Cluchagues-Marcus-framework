# Cluchagues-Marcus Framework (in silico)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23242640.svg)](https://doi.org/10.5281/zenodo.23242640)

Code accompanying the manuscript *"A Physics-Constrained Marcus-Theory Optimization Framework Restores Parameter Identifiability in Deep-Pocket Oncogenic Drug Kinetics: A KRAS G12C and STAT3 Case Study"*.

The code couples non-adiabatic Marcus kinetics with a Hill-type dose-response equation derived from a steady-state scheme, simulates synthetic data, fits them by trust-region-reflective least squares, assesses parameter identifiability from Jacobian singular values, and tests the identifiability result on real napabucasin cell-viability curves.

> **Scope.** The simulations use synthetic data generated from the fitted model. The real napabucasin curves are cell-viability data and serve only as a numerical stress test of fitting and identifiability; they do not test the dielectric mechanism or binding kinetics. Several parameters (ΔG°, k_off,q, n, the calibration λ_ext = 0.46 eV) are assumptions, listed in Table 2 of the manuscript and at the top of the script.

## Contents

| File | Description |
|---|---|
| `Supplementary_Code_S1.py` | Complete analysis: simulation, 200 noise replicates, three fitting configurations, propositions checks, sensitivity analysis, two case studies, real-data test, Figures 1-3 |
| `pocket_radius.py` | Measures a pocket radius from a PDB entry (needs internet access) |
| `dose-data_napabucasin.csv` | Napabucasin dose-response curves used in the real-data test (source: see manuscript, Section 2.4) |
| `README.md`, `LICENSE`, `CITATION.cff` | Documentation, MIT licence, citation metadata |

## Requirements

Developed and tested with Python 3.12.3 on Ubuntu 24.04.4 LTS, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8 and pandas.

```bash
pip install numpy==2.4.4 scipy==1.17.1 matplotlib==3.10.8 pandas
```

Other recent versions should work, but the last digits may differ.

## Run

```bash
python Supplementary_Code_S1.py
```

Keep `dose-data_napabucasin.csv` in the same folder (the real-data section is skipped if it is absent). Random numbers use `numpy.random.default_rng` with fixed seeds (0-199 for replicates, 42 for the representative data set, 2026 for the sensitivity analysis). Outputs: `Figure1.png/.pdf`, `Figure2.png/.pdf`, `Figure3.png/.pdf`, `results.json`, `sensitivity.json`.

## Pocket radius

```bash
python pocket_radius.py 6UT0 6NJS --json pocket_radius_results.json
```

The script downloads each entry from RCSB, takes the largest ligand (or `--lig`), and reports the median distance from the ligand centroid to the protein heavy atoms within 4.5 Å of the ligand (`--cutoff`, `--chain` can be changed). Values used in the manuscript (chain A, 4.5 Å):

| Entry | Ligand | Lining atoms | R_median | R_max |
|---|---|---|---|---|
| 6UT0 (K-Ras G12C + adagrasib) | M1X | 86 | 0.72 nm | 1.22 nm |
| 6NJS (STAT3 core + SD-36) | KQV | 71 | 0.83 nm | 1.65 nm |

Reference constants: adagrasib K_I = 1.43 µM (mass spectrometry, FDA review NDA 216340; covalent K_I, an upper bound of the reversible K_d) and SD-36 K_i = 11 nM (fluorescence polarization, Bai et al. 2019, *Cancer Cell*, https://doi.org/10.1016/j.ccell.2019.10.002).

## Expected key results

If the run reproduces the manuscript, `results.json` should show approximately:

| Quantity | Value |
|---|---|
| λ_ext, KRAS case, unshielded → shielded | 0.46 → 0.047 eV (-89.8%) |
| k_C ratio at ΔG° = -0.10 eV | 8.0-fold (KRAS), 3.6-fold (STAT3) |
| EC50 shift | 2.6-fold (KRAS), 1.9-fold (STAT3) |
| Acceleration threshold √(λ_s λ_u) | 0.147 eV (KRAS), 0.128 eV (STAT3) |
| Null dimension (ν_eff free / ν_eff fixed / ν_eff and K_d fixed) | 2 / 1 / 0 |
| Median r², simulated data (200 replicates) | 0.987-0.988 |
| Real data (701 cell lines): median r² of 4PL fits | 0.91 |
| Real data: SSE change when K_d is anchored at 0.3-10 µM | < 10⁻⁷ % (k_off,q shifts 12.6 → 9.1 in log10) |

## Authors and roles (CRediT)

- **L. M. Cluchagues Abam** (corresponding author): Conceptualization, Methodology, Writing - original draft, Project administration
- **I. H. Mfabo Kameni**: Validation, Formal analysis (review of the code and results), Writing - review & editing
- **F. Eya'ane Meva**: Supervision, Writing - review & editing

## Use of AI

The code, numerical analysis and figures were generated and executed with the assistance of **Claude Sonnet 5.5** (Anthropic; model identifier `claude-sonnet-5-5`), accessed through the Claude chat application in October 2026 with its code-execution tool. The model is not an author. The authors re-ran the code and take responsibility for the content.

## How to cite

Cite the archived release (use the concept DOI above for "all versions", or the version DOI of the release you used):

> Cluchagues Abam, L. M., Mfabo Kameni, I. H., & Eya'ane Meva, F. (2026). *Cluchagues-Marcus framework (in silico validation)* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23242640

and the associated manuscript once published.

## Licence

MIT License (see `LICENSE`).

## Contact

L. M. Cluchagues Abam, University of Douala, Cameroon.
