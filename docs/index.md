# Timber

**Compile classical ML models to native C. Serve them in microseconds.**

[![CI](https://github.com/kossisoroyce/timber/actions/workflows/ci.yml/badge.svg)](https://github.com/kossisoroyce/timber/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/timber-compiler.svg)](https://pypi.org/project/timber-compiler/)
[![Python](https://img.shields.io/pypi/pyversions/timber-compiler.svg)](https://pypi.org/project/timber-compiler/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/kossisoroyce/timber/blob/main/LICENSE)

---

Timber is a **classical ML inference compiler**. It takes a trained ML model — XGBoost,
LightGBM, scikit-learn (trees, linear, SVM, OneClassSVM, IsolationForest, NaiveBayes,
k-NN, GPR), CatBoost, ONNX, or a URDF robot description — and compiles it into a
self-contained C99 shared library. A built-in HTTP server (Ollama-compatible) serves
the compiled model immediately.

For hardware-accelerated, safety-critical, or embedded deployments, the bundled
`timber-accel` CLI emits AVX2/AVX-512/NEON/SVE/RVV SIMD, CUDA/Metal/OpenCL GPU,
Xilinx/Intel FPGA HLS, and Cortex-M/ESP32/STM32 variants — plus WCET analysis,
DO-178C / ISO 26262 / IEC 62304 certification reports, Ed25519 artifact signing,
AES-256-GCM encryption, and ROS 2 / PX4 / gRPC server generators. **Everything ships in
one `pip install`.**

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
| [Accel Guide](accel.md) | SIMD / GPU / HLS / embedded backends, WCET, certification, signing, deployment |
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
| scikit-learn | `.pkl` | GBT, RandomForest, ExtraTrees, Linear/Logistic, SVM, **OneClassSVM**, **IsolationForest**, **GaussianNB**, **k-NN**, **GPR**, Pipeline |
| CatBoost | `.json` | JSON export only |
| ONNX | `.onnx` | TreeEnsemble, LinearClassifier/Regressor, SVMClassifier/Regressor, Normalizer, Scaler |
| URDF | `.urdf` | Robot forward kinematics → 4×4 homogeneous transform |

## Output targets

| Target | Description |
|--------|-------------|
| **C99** | Shared library (`.so`) or static library (`.a`) for servers and embedded systems |
| **WebAssembly** | `.wat` + JS bindings for browser and edge deployment |
| **MISRA-C:2012** | Safety-critical C with built-in 8-rule compliance checker |
| **LLVM IR** | `.ll` text IR with configurable target triple (x86_64, aarch64, Cortex-M, RISC-V) |
| **SIMD C** | AVX2, AVX-512, ARM NEON, ARM SVE, RISC-V V (RVV) — via `timber-accel compile` |
| **GPU** | CUDA (SM 7.5 / 8.6), Apple Metal, OpenCL — via `timber-accel compile` |
| **FPGA HLS** | Xilinx Vitis HLS, Intel FPGA SDK (OpenCL) |
| **Embedded** | Cortex-M4/M7 static-buffer C, ESP32, STM32 |

## New in v0.6.0

- **TimberAccelerate merged into core** — `timber.accel` package + `timber-accel` CLI ship in every install
- **SIMD codegen** — AVX2, AVX-512, NEON, SVE, RVV emitters
- **GPU codegen** — CUDA, Metal, OpenCL
- **FPGA HLS** — Xilinx Vitis, Intel FPGA SDK
- **Embedded** — Cortex-M / ESP32 / STM32 no-heap emitters
- **WCET analysis** — per-stage cycle counts for 5 architectures with advisory disclaimer
- **Certification reports** — DO-178C, ISO 26262, IEC 62304
- **Supply chain** — Ed25519 signing, AES-256-GCM encryption, TPM hooks, air-gapped bundles
- **Deployment generators** — ROS 2 node, PX4 module, C++ gRPC server
- 650+ tests passing

## New in v0.5.0

- 5 new sklearn primitives: IsolationForest, OneClassSVM, GaussianNB, GPR, k-NN
- URDF forward-kinematics frontend — compile any URDF to a C FK function
- `timber serve robot.urdf` auto-detects and serves kinematics

## License

Apache-2.0 — [view full text](https://github.com/kossisoroyce/timber/blob/main/LICENSE)
