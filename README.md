# Timber

<p align="center">
  <strong>Compile classical ML models to native C. Serve them in microseconds.</strong>
</p>

<p align="center">
  <a href="https://github.com/kossisoroyce/timber/actions/workflows/ci.yml"><img src="https://github.com/kossisoroyce/timber/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
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
  <a href="paper/timber_paper.pdf">Technical Paper</a> ·
  <a href="skill.md">Agent Skill</a>
</p>

---

Timber takes a trained ML model — XGBoost, LightGBM, scikit-learn, CatBoost, ONNX (tree ensembles, linear models, SVMs), or a **URDF robot description** — runs it through a multi-pass optimizing compiler, and emits a **self-contained C99 inference artifact** with zero runtime dependencies. A built-in HTTP server (Ollama-compatible API) lets you serve any model — local file or remote URL — in one command.

> **~2 µs single-sample inference · ~336× faster than Python XGBoost · ~48 KB artifact · zero runtime dependencies**

---

## See it in action

> **Point Timber at any URL and it downloads, compiles, and serves in one command — no setup required.**

```console
$ pip install timber-compiler
$ timber serve https://raw.githubusercontent.com/kossisoroyce/timber/main/examples/breast_cancer_model.json

  ✓ Downloaded              breast_cancer_model.json
  ✓ Format detected         xgboost
  ✓ Parsed model            50 trees · 30 features · binary:logistic
  ✓ Optimized               3/5 passes applied
  ✓ Generated C99           169 lines
  ✓ Compiled binary         47.9 KB

  Serving    breast_cancer_model
  Endpoint   http://localhost:11434
  Framework  xgboost  ·  50 trees  ·  30 features

  POST  http://localhost:11434/api/predict
  GET   http://localhost:11434/api/models
  GET   http://localhost:11434/api/health
```

**Predict immediately:**

```console
$ curl -s http://localhost:11434/api/predict \
    -H 'Content-Type: application/json' \
    -d '{"model": "breast_cancer_model", "inputs": [[1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]}'

{"model": "breast_cancer_model", "outputs": [[0.9971]], "n_samples": 1}
```

**Or load from a local file and serve by name:**

```console
$ timber load fraud_model.json --name fraud-detector
$ timber serve fraud-detector
```

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

**Serve any model directly from a URL — no pre-download step:**

```bash
timber serve https://yourhost.com/models/fraud_model.json
```

**Or load a local model and serve by name:**

```bash
timber load fraud_model.json --name fraud-detector
timber serve fraud-detector
```

**Predict:**

```bash
curl -s http://localhost:11434/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "fraud-detector", "inputs": [[1.2, 0.4, 3.1, 0.9]]}'
```

```json
{"model": "fraud-detector", "outputs": [[0.031]], "latency_us": 1.8}
```

**That's it.** No model server configuration, no Python runtime in the hot path.

**Or serve a robot's forward kinematics from a URDF file:**

```bash
timber serve robot.urdf
```

```bash
curl -s http://localhost:11434/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "robot", "inputs": [[0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7]]}'
```

```json
{"model": "robot", "outputs": [[0.082, -0.982, 0.170, 0.310, 0.959, 0.031, -0.283, 0.053, 0.272, 0.187, 0.944, 1.180, 0.0, 0.0, 0.0, 1.0]], "latency_us": 65.0}
```

---

## Supported Formats

| Framework | File format | Notes |
|-----------|-------------|-------|
| XGBoost | `.json` | All objectives; multiclass, binary, regression; XGBoost 3.1+ per-class base_score |
| LightGBM | `.txt`, `.model`, `.lgb` | All objectives including multiclass |
| scikit-learn | `.pkl`, `.pickle` | GradientBoostingClassifier/Regressor, RandomForest, ExtraTrees, DecisionTree, Pipeline |
| ONNX | `.onnx` | `TreeEnsembleClassifier/Regressor`, `LinearClassifier/Regressor`, `SVMClassifier/Regressor`, `Normalizer`, `Scaler` |
| CatBoost | `.json` | JSON export (`save_model(..., format='json')`) |
| URDF | `.urdf` | Robot description → forward kinematics; outputs 4×4 homogeneous transform; inputs are joint angles |

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
| **Formats** | 5 (trees + linear + SVM) | Each framework only | ONNX only | GBDTs | LightGBM only |
| **C export** | Yes (C99) | No | No | Yes | No |
| **LLVM IR export** | Yes | No | No | No | No |
| **Edge / embedded** | Yes (Cortex-M4/M33, RISC-V) | No | Partial | Partial | No |
| **MISRA-C output** | Yes | No | No | No | No |
| **Differential privacy** | Yes | No | No | No | No |

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

- **ONNX** — supports `TreeEnsemble`, `LinearClassifier/Regressor`, `SVMClassifier/Regressor`, `Normalizer`, `Scaler`; other operators (e.g., neural network layers) are not yet supported
- **CatBoost** — requires JSON export (`save_model(..., format='json')`); native binary format not supported
- **scikit-learn** — major estimators and `Pipeline` wrappers are supported; uncommon custom estimators may require a custom front-end
- **Pickle** — follow standard pickle security hygiene; only load artifacts from trusted sources
- **XGBoost** — JSON model format is the primary path; binary booster format is not supported
- **LLVM IR** — currently emitted as text (`.ll`); requires a local LLVM/Clang installation to produce native code from it
- **MISRA-C** — the built-in compliance checker covers the rules most relevant to generated code; it is not a substitute for a certified static analysis tool

---

## Roadmap

| Status | Item |
|--------|------|
| ✅ | XGBoost, LightGBM, scikit-learn, CatBoost, ONNX front-ends |
| ✅ | Multi-pass IR optimizer (dead-leaf, quantization, branch sort, scaler fusion) |
| ✅ | C99 emitter with WebAssembly target |
| ✅ | Ollama-compatible HTTP inference server with multi-worker FastAPI |
| ✅ | PyPI packaging with OIDC trusted publishing |
| ✅ | ONNX Linear/SVM/Normalizer/Scaler operator support |
| ✅ | ARM Cortex-M4/M33 and RISC-V rv32imf/rv64gc embedded deployment profiles |
| ✅ | MISRA-C:2012 compliant output mode with built-in compliance checker |
| ✅ | LLVM IR backend with configurable target triples |
| ✅ | Differential privacy (Laplace + Gaussian) inference mode |
| ✅ | Richer `bench` reports: P50/P95/P99/P999, CV%, JSON + HTML output |
| 🔄 | Remote model registry (`timber pull` from hosted model library) |
| 🔲 | Neural network operator support (MLPClassifier) |
| 🔲 | ONNX export path (Timber IR → ONNX) |
| 🔲 | Rust backend emitter |

---

## Development

```bash
git clone https://github.com/kossisoroyce/timber.git
cd timber
pip install -e ".[dev]"
pytest tests/ -v                    # 436 tests
ruff check timber/                  # linting
```

The test suite covers: parsers (sklearn, ONNX, XGBoost, LightGBM, CatBoost), IR layer (serialization, deep_copy, all stage types), optimizer passes (correctness, idempotency, pipeline fusion math), C99/WASM/MISRA-C/LLVM IR emitters (compile + numeric accuracy), differential privacy (statistical correctness, all dtypes), and full end-to-end pipelines.

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
