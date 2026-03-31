#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/runs

echo "== SCALA1D phenomenological: low p =="
./build/ca_run \
  --model scala1d \
  --noise-model phenomenological \
  --distance 15 \
  --samples 100 \
  --max-steps 100000 \
  --reset-time 16 \
  --p-data 0.005 \
  --p-meas 0.0 \
  --p-signal 0.0 \
  --seed 12345 \
  --output results/runs/scala1d_pheno_lowp.npy

echo "== SCALA1D phenomenological: high p =="
./build/ca_run \
  --model scala1d \
  --noise-model phenomenological \
  --distance 15 \
  --samples 100 \
  --max-steps 100000 \
  --reset-time 16 \
  --p-data 0.05 \
  --p-meas 0.0 \
  --p-signal 0.0 \
  --seed 12345 \
  --output results/runs/scala1d_pheno_highp.npy

echo "== SCALA1D code capacity: low p =="
./build/ca_run \
  --model scala1d \
  --noise-model code_capacity \
  --distance 15 \
  --samples 5000 \
  --max-steps 1000 \
  --p-data 0.1 \
  --seed 12345 \
  --output results/runs/scala1d_cc_lowp.npy

echo "== SCALA1D code capacity: high p =="
./build/ca_run \
  --model scala1d \
  --noise-model code_capacity \
  --distance 15 \
  --samples 5000 \
  --max-steps 1000 \
  --p-data 0.3 \
  --seed 12345 \
  --output results/runs/scala1d_cc_highp.npy

echo "Done."
