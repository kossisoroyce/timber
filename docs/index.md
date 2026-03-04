# Timber

**Compile classical ML models to native C. Serve them in microseconds.**

[![CI](https://github.com/kossisoroyce/timber/actions/workflows/ci.yml/badge.svg)](https://github.com/kossisoroyce/timber/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/timber-compiler.svg)](https://pypi.org/project/timber-compiler/)
[![Python](https://img.shields.io/pypi/pyversions/timber-compiler.svg)](https://pypi.org/project/timber-compiler/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/kossisoroyce/timber/blob/main/LICENSE)

---

Timber is a **classical ML inference compiler**. It takes a trained ML model — XGBoost,
LightGBM, scikit-learn, CatBoost, or ONNX (trees, linear models, SVMs) — and compiles it
into a self-contained C99 shared library. A built-in HTTP server (Ollama-compatible) serves
the compiled model immediately.

> **~2 µs latency · ~336× faster than Python XGBoost · ~48 KB artifact · zero runtime dependencies**

## In 3 commands

```bash
pip install timber-compiler
timber load model.json --name my-model
timber serve my-model
```

Then predict:

```bash
curl http://localhost:11434/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"model": "my-model", "inputs": [[1.2, 0.4, 3.1, ...]]}'
```

## How it works

```
  Model file → Parser → TimberIR → Optimizer → C99 Emitter → gcc → .so → HTTP
```

1. **Parse** — reads any supported format into a framework-agnostic IR
2. **Optimize** — 6 passes: dead-leaf elimination, scaler fusion, threshold quantization, branch sorting
3. **Emit** — generates portable C99 / LLVM IR / WebAssembly / MISRA-C (no heap, no recursion)
4. **Compile** — `gcc`/`clang` produces a shared library in ~50 ms
5. **Serve** — Ollama-compatible HTTP API with multi-worker FastAPI + `GET /api/metrics`

## Navigation

| Section | What's in it |
|---------|-------------|
| [Getting Started](getting-started.md) | Installation, first model, 60-second quickstart |
| [Examples](examples.md) | Per-framework code examples (XGBoost, LightGBM, sklearn, CatBoost, ONNX) |
| [Advanced Usage](advanced.md) | Direct compilation, C embedding, WASM, embedded targets, LLVM IR, differential privacy |
| [API Reference](api-reference.md) | Full CLI, HTTP API, Python API, C API, and Privacy API |
| [IR Reference](ir-reference.md) | TimberIR type system for contributors and tool builders |
| [Architecture](architecture.md) | Compiler internals — frontend, optimizer, backend, store |
| [FAQ](faq.md) | Common questions answered |
| [Changelog](../CHANGELOG.md) | Release history |

## Supported frameworks

| Framework | File | Notes |
|-----------|------|-------|
| XGBoost | `.json` | All objectives; XGBoost 3.1+ per-class base_score handled |
| LightGBM | `.txt` / `.model` | All objectives |
| scikit-learn | `.pkl` | GBT, RandomForest, ExtraTrees, Pipeline |
| CatBoost | `.json` | JSON export only |
| ONNX | `.onnx` | TreeEnsemble, LinearClassifier/Regressor, SVMClassifier/Regressor, Normalizer, Scaler |

## Output targets

| Target | Description |
|--------|-------------|
| **C99** | Shared library (`.so`) or static library (`.a`) for servers and embedded systems |
| **WebAssembly** | `.wat` + JS bindings for browser and edge deployment |
| **MISRA-C:2012** | Safety-critical C with built-in 8-rule compliance checker |
| **LLVM IR** | `.ll` text IR with configurable target triple (x86_64, aarch64, Cortex-M, RISC-V) |

## New in v0.4.0

- ONNX Linear / SVM / Normalizer / Scaler operator support
- Embedded cross-compilation profiles: Cortex-M4, Cortex-M33, RISC-V rv32imf/rv64gc
- LLVM IR backend with multi-target support
- Differential privacy module (Laplace + Gaussian mechanisms)
- Enhanced `bench`: P50/P95/P99/P999, CV%, JSON + HTML reports
- 3 ONNX parser bug fixes (attribute names, binary weight rows, C99 buffer guard)
- 436 tests passing

## License

Apache-2.0 — [view full text](https://github.com/kossisoroyce/timber/blob/main/LICENSE)
