# Timber

<p align="center">
  <strong>Compile classical ML models to native C. Serve them in microseconds.</strong>
</p>

<p align="center">
  <a href="https://github.com/kossisoroyce/timber/actions/workflows/ci.yml"><img src="https://github.com/kossisoroyce/timber/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/timber-compiler/"><img src="https://img.shields.io/pypi/v/timber-compiler.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/timber-compiler/"><img src="https://img.shields.io/pypi/pyversions/timber-compiler.svg" alt="Python versions"></a>
  <a href="https://pypi.org/project/timber-compiler/"><img src="https://img.shields.io/pypi/dm/timber-compiler.svg" alt="Monthly downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://codecov.io/gh/kossisoroyce/timber"><img src="https://codecov.io/gh/kossisoroyce/timber/branch/main/graph/badge.svg" alt="Coverage"></a>
</p>

<p align="center">
  <a href="https://kossisoroyce.github.io/timber/">Documentation</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://pypi.org/project/timber-compiler/">PyPI</a> ·
  <a href="paper/timber_paper.pdf">Technical Paper</a>
</p>

---

Timber takes a trained tree-based model — XGBoost, LightGBM, scikit-learn, CatBoost, or ONNX — runs it through a multi-pass optimizing compiler, and emits a **self-contained C99 inference binary** with zero runtime dependencies. A built-in HTTP server (Ollama-compatible API) lets you serve the compiled model immediately.

> **~2 µs single-sample inference · ~336× faster than Python XGBoost · ~48 KB artifact · zero runtime dependencies**

---

## Table of Contents

- [Who is this for?](#who-is-this-for)
- [How it works](#how-it-works)
- [Quick Start](#quick-start)
- [Supported Formats](#supported-formats)
- [Performance](#performance)
- [Runtime Comparison](#runtime-comparison)
- [API Reference](#api-reference)
- [CLI Reference](#cli-reference)
- [Examples](#examples)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Development](#development)
- [Citation](#citation)
- [Community & Governance](#community--governance)
- [License](#license)

---

## Who is this for?

Timber is built for teams that need **fast, predictable, and portable inference**:

- **Fraud & risk teams** — run classical models in sub-millisecond transaction paths without Python overhead
- **Edge & IoT deployments** — ship a ~48 KB C artifact to gateways, microcontrollers, or ARM Cortex-M targets
- **Regulated industries** — finance, healthcare, and automotive teams that need deterministic, auditable inference artifacts
- **Platform & infra teams** — eliminate the Python model-serving stack from your critical path entirely

---

## How it works

```
  ┌─────────────────────────────────────────────────────────┐
  │                     timber load                         │
  │                                                         │
  │  Model file  ──►  Parser  ──►  Timber IR  ──►  Optimizer│
  │  (.json/.pkl/                  (typed AST)   (dead-leaf  │
  │   .txt/.onnx)                               elim, quant, │
  │                                              branch-sort) │
  │                                     │                    │
  │                                     ▼                    │
  │                               C99 Emitter                │
  │                                     │                    │
  │                    ┌────────────────┼────────────────┐   │
  │                    ▼                ▼                ▼   │
  │               model.c         model.h        model_data.c│
  │               (inference)     (public API)   (tree data)  │
  │                    │                                     │
  │                    └──► gcc / clang ──► model.so         │
  └─────────────────────────────────────────────────────────┘
                              │
                              ▼
                      timber serve <name>
                   http://localhost:11434/api/predict
```

The compiler pipeline:
1. **Parse** — reads the native model format into a framework-agnostic Timber IR
2. **Optimize** — dead-leaf elimination, threshold quantization, constant-feature folding, branch sorting
3. **Emit** — generates deterministic, portable C99 with no dynamic allocation and no recursion
4. **Compile** — `gcc`/`clang` produces a shared library loaded via `ctypes`
5. **Serve** — an Ollama-compatible HTTP API wraps the binary for drop-in integration

---

## Quick Start

```bash
pip install timber-compiler
```

```bash
# Load and compile a model (auto-detects format)
timber load fraud_model.json --name fraud-detector

# Start serving (Ollama-compatible endpoint)
timber serve fraud-detector
```

```bash
# Predict
curl -s http://localhost:11434/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "fraud-detector", "inputs": [[1.2, 0.4, 3.1, 0.9]]}'
```

```json
{"model": "fraud-detector", "outputs": [[0.031]], "latency_us": 1.8}
```

**That's it.** No model server configuration, no Python runtime in the hot path.

---

## Supported Formats

| Framework | File format | Notes |
|-----------|-------------|-------|
| XGBoost | `.json` | All objectives; multiclass, binary, regression |
| LightGBM | `.txt`, `.model`, `.lgb` | All objectives including multiclass |
| scikit-learn | `.pkl`, `.pickle` | GradientBoostingClassifier/Regressor, RandomForest, ExtraTrees, DecisionTree, Pipeline |
| ONNX | `.onnx` | `TreeEnsembleClassifier` and `TreeEnsembleRegressor` ML operators |
| CatBoost | `.json` | JSON export (`save_model(..., format='json')`) |

---

## Performance

> Benchmarks run on Apple M2 Pro · 16 GB RAM · macOS · XGBoost binary classifier · 50 trees · max depth 4 · 30 features (sklearn `breast_cancer`) · 10,000 timed iterations after 1,000 warmup.

| Runtime | Single-sample latency | Throughput | Speedup vs Python |
|---------|----------------------|------------|-------------------|
| **Timber (native C)** | **~2 µs** | **~500,000 / sec** | **336×** |
| ONNX Runtime | ~80–150 µs | ~10,000 / sec | ~5× |
| Treelite (compiled) | ~10–30 µs | ~50,000 / sec | ~20× |
| Python XGBoost | ~670 µs | ~1,500 / sec | 1× (baseline) |
| Python scikit-learn | ~900 µs | ~1,100 / sec | 0.7× |

Latency is **in-process** (not HTTP round-trip). Network overhead adds ~50–200 µs depending on your stack.

### Reproduce these numbers

```bash
python benchmarks/run_benchmarks.py --output benchmarks/results.json
python benchmarks/render_table.py   --input  benchmarks/results.json
```

See [`benchmarks/`](benchmarks/) for full methodology, hardware capture script, and optional ONNX Runtime / Treelite / lleaves comparisons.

---

## Runtime Comparison

| | Timber | Python serving | ONNX Runtime | Treelite | lleaves |
|---|---|---|---|---|---|
| **Latency** | ~2 µs | 100s of µs–ms | ~100 µs | ~10–30 µs | ~50 µs |
| **Runtime deps** | None | Python + framework | ONNX Runtime libs | Treelite runtime | Python + LightGBM |
| **Artifact size** | ~48 KB | 50–200+ MB process | MBs | MB-scale | Python env |
| **Formats** | 5 | Each framework only | ONNX only | GBDTs | LightGBM only |
| **C export** | Yes (C99) | No | No | Yes | No |
| **Edge / embedded** | Yes | No | Partial | Partial | No |
| **Audit / MISRA** | Roadmap | No | No | No | No |

---

## API Reference

Timber's server exposes an **Ollama-compatible REST API** on `http://localhost:11434` by default.

| Endpoint | Method | Body / Params | Description |
|----------|--------|---------------|-------------|
| `/api/predict` | POST | `{"model": str, "inputs": [[float]]}` | Run inference |
| `/api/generate` | POST | same as `/api/predict` | Ollama alias |
| `/api/models` | GET | — | List all loaded models |
| `/api/model/:name` | GET | — | Model metadata & schema |
| `/api/health` | GET | — | Health check |

**Example — batch inference:**

```bash
curl -s http://localhost:11434/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fraud-detector",
    "inputs": [
      [1.2, 0.4, 3.1, 0.9],
      [0.1, 2.3, 1.0, 4.4]
    ]
  }'
```

---

## CLI Reference

```
timber load   <path> --name <name>   Compile and register a model
timber serve  <name> [--port N]      Start the inference server
timber list                          List registered models
timber inspect <name>                Show model IR summary and schema
timber validate <name>               Run numerical validation against source
timber bench  <name>                 Benchmark latency and throughput
timber pull   <url>  --name <name>   Download and compile from URL
timber remove <name>                 Remove a model from the registry
```

---

## Examples

Runnable end-to-end examples live in [`examples/`](examples/):

```bash
python examples/quickstart_xgboost.py   # trains, compiles, and benchmarks
python examples/quickstart_lightgbm.py
python examples/quickstart_sklearn.py
```

Each script trains a model, saves it, runs `timber load`, and validates predictions against the source framework.

---

## Limitations

- **ONNX** — currently supports `TreeEnsembleClassifier` / `TreeEnsembleRegressor` operators only
- **CatBoost** — requires JSON export (`save_model(..., format='json')`); native binary format not supported
- **scikit-learn** — major estimators and `Pipeline` wrappers are supported; uncommon custom estimators may require a custom front-end
- **Pickle** — follow standard pickle security hygiene; only load artifacts from trusted sources
- **XGBoost** — JSON model format is the primary path; binary booster format is not supported
- **MISRA-C / safety certification** — deterministic output is guaranteed but formal MISRA-C compliance is on the roadmap, not yet certified

---

## Roadmap

| Status | Item |
|--------|------|
| ✅ | XGBoost, LightGBM, scikit-learn, CatBoost, ONNX front-ends |
| ✅ | Multi-pass IR optimizer (dead-leaf, quantization, branch sort, scaler fusion) |
| ✅ | C99 emitter with WebAssembly target |
| ✅ | Ollama-compatible HTTP inference server |
| ✅ | PyPI packaging with OIDC trusted publishing |
| 🔄 | Remote model registry (`timber pull` from hosted model library) |
| 🔄 | Broader ONNX operator support (linear, SVM, normalizers) |
| 🔄 | ARM Cortex-M / RISC-V embedded deployment profiles |
| 🔄 | MISRA-C compliant output mode for automotive/aerospace |
| 🔄 | Richer benchmark matrices and public reproducibility report |
| 🔲 | LLVM IR target for hardware-specific optimization |
| 🔲 | Differential privacy inference mode |

---

## Development

```bash
git clone https://github.com/kossisoroyce/timber.git
cd timber
pip install -e ".[dev]"
pytest tests/ -v                    # 146 tests
ruff check timber/                  # linting
```

The test suite covers parsers, IR, optimizer passes, C99 emission, WebAssembly emission, numerical accuracy (± 1e-4), and end-to-end compilation for all supported frameworks.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full development guide.

---

## Citation

If you use Timber in research or production, please cite the accompanying technical paper:

```bibtex
@misc{royce2026timber,
  title        = {Timber: Compiling Classical Machine Learning Models to Native Inference Binaries},
  author       = {Kossiso Royce},
  year         = {2026},
  howpublished = {GitHub repository and technical paper},
  institution  = {Electricsheep Africa},
  url          = {https://github.com/kossisoroyce/timber}
}
```

The full paper is available at [`paper/timber_paper.pdf`](paper/timber_paper.pdf).

---

## Community & Governance

- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Code of conduct:** [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- **Security policy:** [`SECURITY.md`](SECURITY.md)
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md)

Bugs and feature requests: [open an issue](https://github.com/kossisoroyce/timber/issues). Questions: [start a discussion](https://github.com/kossisoroyce/timber/discussions).

---

## Support the Project

Timber is developed and maintained by [Electricsheep Africa](https://github.com/electricsheepafrica). If Timber saves your team engineering time, consider supporting continued development:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-%23FFDD00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/electricsheepafrica)

We're also building a hosted library of compiled models — if you have high-performance models you'd like to contribute, [get in touch](https://github.com/kossisoroyce/timber/discussions).

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE) for the full text.
