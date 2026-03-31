# SCALA / Harrington Cellular Automaton Decoders (1D & 2D)

This repository implements and reproduces results for local cellular automaton (CA) decoders for repetition and toric codes:

- **SCALA1D**
- **Harrington1D**
- **SCALA2D**
- **Harrington2D**

Both **code capacity** and **phenomenological noise models** are supported.

The goal of this repository is **reproducibility of results**, not performance optimization.

---

# Repository Structure

```text
cpp/
  include/
  src/
    models/        # CA implementations
    experiments/   # 1D experiments
    bindings/      # pybind11 modules (2D)
    main.cpp       # CLI entrypoint

python/
  experiments/     # 2D experiment runners
  tests/           # sanity tests

scripts/
  scan.py
  summarize_scan.py
  plot_*.py

configs/
  *.json           # scan configurations

results/
  scans/
  figures/
```

---

# Requirements

- C++20 compiler
- Python >= 3.10
- pybind11
- numpy, pandas, matplotlib
- pymatching (for phenomenological decoding)

---

# Build

## 1D executable

```bash
make
```

## 2D modules

```bash
make scala2d
make harrington2d
```

---

# Running Experiments

## 1D models (C++)

Example: SCALA1D, phenomenological

```bash
./build/ca_run   --model scala1d   --noise-model phenomenological   --distance 9   --samples 1000   --max-steps 1000   --reset-time 5   --p-data 0.01   --p-meas 0.0   --p-signal 0.0   --seed 12345   --output results/run.npy
```

Example: Harrington1D, code capacity

```bash
./build/ca_run   --model harrington1d   --noise-model code_capacity   --distance 9   --samples 1000   --max-steps 1000   --p-data 0.1   --U 16   --fN 0.4   --fC 0.9   --seed 12345   --output results/run.npy
```

---

## 2D models (Python)

Example: SCALA2D, phenomenological

```bash
python -m python.experiments.run_scala2d   --noise-model phenomenological   --distance 9   --p-data 0.01   --p-meas 0.01   --p-signal 0.01   --reset-time 5   --samples 100   --seed 12345   --output-dir results/scans/scala2d
```

Example: Harrington2D, code capacity

```bash
python -m python.experiments.run_harrington2d   --noise-model code_capacity   --distance 9   --p-data 0.1   --samples 100   --seed 12345   --output-dir results/scans/harrington2d
```

---

# Parameter Scans

All scans are defined via JSON configs in `configs/`.

Example:

```bash
python scripts/scan.py   --config configs/scala2d_pheno_equal_scan.json
```

---

## Linked Parameters

Some scans require parameters to be linked, for example

- `p_data = p_meas = p_signal`

This is implemented via:

```json
"linked": [
  ["p_data", "p_meas", "p_signal"]
]
```

instead of a full Cartesian product.

---

## Reset-Time Handling

For phenomenological SCALA scans, reset times larger than the code distance are invalid.

The repository handles this in two ways:

- **manual runs**: invalid reset times are rejected immediately
- **scan runs**: invalid jobs are skipped automatically

This avoids wasting time on runs with `reset_time > distance`.

---

# Typical Workflows

## 1D scan

```bash
python scripts/scan.py --config configs/scala1d_pheno_small_scan.json
python scripts/summarize_scan.py   --input-dir results/scans/scala1d_pheno_small   --output-csv results/scans/scala1d_pheno_small_summary.csv
python scripts/plot_tf_vs_p.py   --scala-csv results/scans/scala1d_pheno_small_summary.csv   --harrington-csv results/scans/harrington1d_pheno_small_summary.csv   --output results/figures/tf_vs_p.png   --logx --logy
```

## 2D scan

```bash
python scripts/scan.py --config configs/scala2d_code_capacity_scan.json
python scripts/summarize_scan.py   --input-dir results/scans/scala2d_cc   --output-csv results/scans/scala2d_cc_summary.csv
python scripts/plot_scala2d_cc_paper.py   --csv results/scans/scala2d_cc_summary.csv   --output results/figures/scala2d_cc_paper.png
```

---

# Reproducing Figures

The general workflow is:

1. **run a scan**
2. **summarize the metadata**
3. **plot the summarized data**

Example:

```bash
# run scan
python scripts/scan.py --config <config.json>

# summarize
python scripts/summarize_scan.py   --input-dir <scan-output-dir>   --output-csv <summary.csv>

# plot
python scripts/plot_<...>.py ...
```

---

## Paper-Style Plots

Dedicated plotting scripts are provided for paper-style figures, including:

- `plot_scala1d_pheno_paper.py`
- `plot_scala2d_pheno_paper.py`
- `plot_scala2d_cc_paper.py`
- `plot_harrington2d.py`

For SCALA phenomenological plots, these scripts select the **best reset time** at fixed `(distance, p)` from the scan output.

---

# Makefile Targets

Useful targets include:

```bash
make
make tests
make scala2d
make harrington2d
make scan-scala2d-cc
make scan-scala2d-pheno
make summarize-scala2d-cc
make summarize-scala2d-pheno
make plot-scala2d
```

Depending on your final `Makefile`, additional convenience targets may be available for the 1D and paper-style plots.

---

# Tests

Minimal sanity tests are included to check:

- monotonicity with respect to noise strength
- improvement with code distance
- constructor/input validity

Examples:

```bash
make tests
make test-scala2d
```

If Python tests are used directly:

```bash
python -m python.tests.test_scala2d_code_capacity
python -m python.tests.test_scala2d_distance_trend
python -m python.tests.test_scala2d_phenomenological
```

---

# Notes on the Models

## SCALA decoders

SCALA decoders are local signal-based cellular automata.

- `SCALA1D` acts on the repetition code
- `SCALA2D` acts on the toric code

## Harrington decoders

Harrington decoders are hierarchical cellular automata with memory.

- `Harrington1D` is implemented directly in C++
- `Harrington2D` is implemented as a native Python module plus Python experiment wrappers

## Code capacity vs phenomenological noise

- **code capacity**:
  - one-shot data noise
  - deterministic decoding thereafter
- **phenomenological**:
  - repeated noise during time evolution
  - stopping condition based on logical failure / mismatch criteria

---

# Scope

This repository is intentionally:

- **minimal**
- **explicit**
- **not optimized**

It is designed for:

- reproducing figures
- validating implementations
- experimenting with local CA decoder variants

It is not intended as a polished general-purpose software package.

---

# License

Add a license here if desired.
