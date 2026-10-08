# Cluchagues-Marcus Framework (in silico)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23242641.svg)](https://doi.org/10.5281/zenodo.23242641)

Code accompanying the manuscript *"Modulating Local Microenvironmental Dielectric Permittivity via Biogenic Sub-Nanometer Vectors to Optimize Drug-Target Kinetic Fitting: The Cluchagues-Marcus Framework*.

The script couples non-adiabatic Marcus kinetics with a modified Hill-type dose–response equation, simulates synthetic data, fits them by trust-region-reflective least squares, and assesses parameter identifiability from the Jacobian singular values.

> **Scope.** All data are synthetic and generated from the fitted model. The results show mathematical properties of the formulation, not biological validity. Several parameters (ΔG°, K_d, k_off,q, n) are illustrative assumptions, listed in Table 2 of the manuscript and at the top of the script.

## Contents

| File | Description |
|---|---|
| `Supplementary_Code_S1.py` | Complete analysis: simulation, 200 noise replicates, three fitting configurations, Figure 1 |
| `README.md` | This file |
| `LICENSE` | Software licence |
| `CITATION.cff` | Citation metadata |

## Requirements

Developed and tested with:

- Python 3.12.3 on Ubuntu 24.04.4 LTS
- NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8

```bash
pip install numpy==2.4.4 scipy==1.17.1 matplotlib==3.10.8
```

Other recent versions should work, but exact numbers may differ in the last digits.

## Run

```bash
python Supplementary_Code_S1.py
```

Random numbers use `numpy.random.default_rng` with fixed seeds (0–199 for the replicates, 42 for the representative data set), so results are reproducible. The run takes a few minutes on a standard laptop.

## Outputs (written to the working directory)

- `Figure1.png` (600 dpi) and `Figure1.pdf` (vector): the four-panel Figure 1
- `results.json`: all numerical results reported in Section 3

## Expected key results

If the run reproduces the manuscript, `results.json` should show approximately:

| Quantity | Value |
|---|---|
| λ_ext, unshielded → shielded | 0.46 → 0.047 eV (−89.8%) |
| k_C ratio at ΔG° = −0.10 eV | 8.0-fold |
| EC50 shift | 2.6-fold |
| Median r² (200 replicates) | 0.988 (IQR 0.986–0.989) |
| Null dimension of the Jacobian: ν_eff free / ν_eff fixed / ν_eff and K_d fixed | 2 / 1 / 0 |
| Function evaluations to convergence (median) | 6–7 |

## Authors and roles (CRediT)

- **L. M. Cluchagues Abam** (corresponding author): Conceptualization, Methodology, Writing – original draft, Project administration
- **I. H. Mfabo Kameni**: Validation, Formal analysis, Writing – review & editing
- **F. Eya'ane Meva**: Supervision, Writing – review & editing

## Use of AI

The code was drafted and executed with the assistance of **Claude Sonnet 5.5** (Anthropic; model identifier `claude-sonnet-5-5`), accessed through the Claude chat application in October 2026 using its code-execution tool. The authors reviewed the code and results and take responsibility for them. The model is not an author.

## How to cite

Please cite the archived release:

> Cluchagues Abam, L. M., Mfabo Kameni, I. H., & Eya'ane Meva, F. (2026). *Cluchagues-Marcus framework (in silico validation)*, version V1.0.0 [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23242641

and the associated manuscript once published.

## Licence

MIT License (see `LICENSE`). Replace this line if you choose a different licence.

## Contact

L. M. Cluchagues Abam, University of Douala, Cameroon.
