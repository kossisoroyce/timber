# Timber Benchmarks (Reproducible)

This folder contains scripts to reproduce the benchmark methodology used in Timber docs and README.

## Methodology

### Hardware

Record your hardware before running:

```bash
python benchmarks/system_info.py
```

Example reference environment (used for README numbers):

- CPU: Apple M2 Pro
- RAM: 16 GB
- OS: macOS 15.x
- Python: 3.11

### Model Spec

All scripts default to:

- Dataset: `sklearn.datasets.load_breast_cancer`
- Model: XGBoost binary classifier
- Trees: 50
- Max depth: 4
- Features: 30
- Warmup: 1,000 iterations
- Timed iterations: 10,000 single-sample predictions

### Baselines

The benchmark runner compares:

1. **Python XGBoost** (`booster.predict`)
2. **Timber native predictor** (`TimberPredictor.from_model`)
3. **ONNX Runtime** *(optional, if installed)*
4. **Treelite runtime** *(optional, if installed)*
5. **lleaves** *(optional, if installed)*

Optional backends are auto-skipped if missing.

## Quick Start

From repo root:

```bash
pip install -e ".[dev]"
# Optional backends
pip install onnxruntime skl2onnx treelite_runtime lleaves

python benchmarks/run_benchmarks.py --output benchmarks/results.json
python benchmarks/render_table.py --input benchmarks/results.json
```

## Output

- `results.json` includes:
  - hardware metadata
  - package versions
  - model spec
  - p50/p95/p99/mean latency (microseconds)
  - throughput (predictions/sec)

## Notes

- Results vary by CPU, compiler, and BLAS backend.
- Always include `results.json` and `python benchmarks/system_info.py` output when sharing numbers.
- The benchmark is **in-process inference latency** (not network/HTTP round-trip).
