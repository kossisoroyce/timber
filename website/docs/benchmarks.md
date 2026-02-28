---
sidebar_position: 4
title: Benchmarks
---

# Benchmarks

This page documents Timber's benchmark methodology and how to reproduce results.

## Reference Claim Context

The commonly cited **336x** speedup is measured as:

- **Baseline:** Python XGBoost (`booster.predict`) single-sample inference
- **Timber path:** `TimberPredictor` calling compiled native artifact
- **Metric:** in-process latency (microseconds), excluding HTTP/network overhead

## Methodology

### Hardware and Environment

Reference setup:

- CPU: Apple M2 Pro
- RAM: 16 GB
- OS: macOS
- Python: 3.11

To record your own hardware metadata:

```bash
python benchmarks/system_info.py
```

### Model Specification

- Framework: XGBoost
- Objective: `binary:logistic`
- Trees: 50
- Max depth: 4
- Features: 30
- Dataset: sklearn `breast_cancer`

### Benchmark Parameters

- Warmup iterations: 1,000
- Timed iterations: 10,000
- Input shape: single sample (`batch=1`)

## Reproducible Scripts

All scripts are in [`benchmarks/`](https://github.com/kossisoroyce/timber/tree/main/benchmarks):

- `run_benchmarks.py` — runs Timber vs Python XGBoost and optional backends
- `render_table.py` — renders markdown comparison table from JSON
- `system_info.py` — captures hardware/software metadata

Run from repo root:

```bash
python benchmarks/run_benchmarks.py --output benchmarks/results.json
python benchmarks/render_table.py --input benchmarks/results.json
```

## Comparison Targets

The benchmark runner includes:

1. Python XGBoost (required)
2. Timber native predictor (required)
3. ONNX Runtime (optional)
4. Treelite runtime (optional)
5. lleaves (optional)

Optional targets are skipped automatically when dependencies are missing.

## Reporting Guidance

When publishing benchmark numbers, include:

- Full hardware metadata (`system_info.py` output)
- Model spec (trees, depth, features)
- Warmup and timed iteration counts
- Baselines used
- Raw `benchmarks/results.json` artifact

This keeps claims auditable and reproducible.
