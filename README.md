# Cluchagues-Marcus Framework (In Silico Validation)

This repository hosts the official computational implementation of the Cluchagues-Marcus Unified Model as described in the manuscript submitted to *In Silico Pharmacology*. 

## Repository Contents
* `Supplementary_Code_S1.py`: Core Python script containing non-adiabatic Marcus kinetics coupled with a modified Hill population-dynamics engine. It executes 200 stochastic replicates via the Trust Region Reflective (TRR) algorithm.

## System Prerequisites
Execution requires a standard Python 3 environment. The codebase has been fully verified using the following execution environment:
* **Operating System:** Ubuntu 24.04.4 LTS (or compatible Linux/macOS/Windows environments)
* **Python Version:** 3.12.3
* **Required Libraries:** `numpy >= 2.4.4`, `scipy >= 1.17.1`, `matplotlib >= 3.10.8`

## Execution Instructions
To run the full optimization suite, cross-verify parameter identifiability statistics, and reproduce the manuscript figures, clone this repository and execute the script from your terminal:

```bash
python Supplementary_Code_S1.py
```

## Generated Outputs
Upon completion, the code will dynamically output the following validation assets into your directory:
1. `Figure1.png`: High-resolution (600 DPI) publication-ready composite figure panel.
2. `Figure1.pdf`: Vector format graphic panel for journal typesetting.
3. `results.json`: Log file containing exact numerical indices, median r² values, function evaluation metrics (\(n_{\rm fev}\)), and Jacobian singular value ranks.
# Cluchagues-Marcus-framework
