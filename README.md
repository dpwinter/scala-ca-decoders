# SCALA / Harrington Cellular Automaton Decoders (1D & 2D)

This repository implements and reproduces results for local cellular automaton (CA) decoders for repetition and toric codes:

- SCALA1D
- Harrington1D
- SCALA2D
- Harrington2D

Both code capacity and phenomenological noise models are supported.

The goal of this repository is reproducibility of results, not performance optimization.

---

# Quick Start

```bash
make
make scala2d

python scripts/scan.py --config configs/scala2d_cc_paper_scan.json

python scripts/summarize_scan.py   --input-dir results/scans/scala2d_cc_paper   --output-csv results/scans/scala2d_cc_paper_summary.csv

python scripts/plot_code_capacity_paper.py   --csv results/scans/scala2d_cc_paper_summary.csv   --output results/figures/scala2d_cc.png
```

---

# Repository Structure

cpp/
  include/
  src/
    models/
    experiments/
    bindings/
    main.cpp

python/
  experiments/
  tests/

scripts/
  scan.py
  summarize_scan.py
  plot_*.py

configs/
  *.json

results/
  scans/
  figures/

---

# Requirements

- C++20 compiler
- Python >= 3.10
- pybind11
- numpy, pandas, matplotlib
- pymatching

---

# Build

## 1D executable

make

## 2D modules

make scala2d
make harrington2d

---

# Running Experiments

## 1D models (C++)

Example:

./build/ca_run \
  --model scala1d \
  --noise-model phenomenological \
  --distance 9 \
  --samples 1000 \
  --max-steps 1000 \
  --reset-time 5 \
  --p-data 0.01 \
  --p-meas 0.0 \
  --p-signal 0.0 \
  --seed 12345 \
  --output results/run.npy

---

## 2D models (Python)

Example:

python -m python.experiments.run_scala2d \
  --noise-model phenomenological \
  --distance 9 \
  --p-data 0.01 \
  --p-meas 0.01 \
  --p-signal 0.01 \
  --reset-time 5 \
  --samples 100 \
  --seed 12345 \
  --output-dir results/scans/scala2d

---

# Parameter Scans

Example:

python scripts/scan.py --config configs/scala2d_pheno_equal_scan.json

Linked parameters example:

"linked": [
  ["p_data", "p_meas", "p_signal"]
]

---

# Reset-Time Handling

For phenomenological SCALA runs:

- reset_time must satisfy reset_time <= distance
- invalid runs are skipped automatically

---

# Workflow

1. Run scan

python scripts/scan.py --config <config.json>

2. Summarize

python scripts/summarize_scan.py \
  --input-dir <scan-output-dir> \
  --output-csv <summary.csv>

3. Plot

python scripts/plot_*.py ...

---

# Plotting

Scripts:

- plot_code_capacity_paper.py
- plot_pheno_two_panel_paper.py
- plot_pheno_equal_paper.py

These:

- load CSV summaries
- compute means and error bars from .npy
- select best reset time (SCALA)

No fitting is performed.

---

# Tests

make tests

or

python -m python.tests.test_scala2d_code_capacity

---

# Scope

This repository is minimal and intended for:

- reproducing figures
- validating implementations
- experimenting with CA rules

---

# License

This project is licensed under the MIT License.  
See the LICENSE file for details.
