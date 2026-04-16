---
sidebar_position: 1
slug: /
title: Introduction
---

# Timber

**ML inference compiler with hardware acceleration.** Compile XGBoost, LightGBM, sklearn, CatBoost, and ONNX to C99, SIMD, GPU, or FPGA — with WCET analysis, safety certification, and supply-chain security built in.

Timber is an ahead-of-time (AOT) compiler that transforms trained machine learning models into optimized, self-contained inference code. No Python runtime at inference time. No dynamic allocation. No recursion. Just fast, auditable, portable code that runs on servers, edge devices, microcontrollers, and safety-critical systems.

## The Problem

The typical production path for an XGBoost or scikit-learn model today:

1. Train model in Python, serialize to `.json` or `.pkl`
2. Wrap in a Flask or FastAPI endpoint
3. Deploy with the full Python runtime
4. Pay the overhead of CPython on **every inference call**

This means a **2 MB model** needs a **200 MB Python process** to serve it, with P99 latencies measured in **milliseconds** rather than microseconds.

## The Solution

Timber treats your trained model as a **program specification** and compiles it:

```
Model Artifact → Parse → IR → Optimize (6 passes) → Emit → C99 → .so → Serve
```

The result:

| Metric | Python Inference | Timber |
|--------|-----------------|--------|
| P50 Latency | 672 µs | **2 µs** (336× faster) |
| P99 Latency | 13,468 µs | **3 µs** (4,489× faster) |
| Throughput | 840/s | **447,868/s** (533× faster) |
| Artifact Size | ~50 MB | **48 KB** (1,000× smaller) |
| Dependencies | Python + NumPy + XGBoost | **None** (C99 only) |

## Ollama-Style Workflow

Inspired by [Ollama](https://ollama.ai)'s developer experience for LLMs, Timber provides the same simplicity for classical ML:

```bash
pip install timber-compiler
timber load model.json --name fraud-detector
timber serve fraud-detector
```

```bash
curl http://localhost:11434/api/predict \
  -d '{"model": "fraud-detector", "inputs": [[1.0, 2.0, ...]]}'
```

## What Timber Supports

### Input Frameworks
- **XGBoost** — JSON model dumps (v2.0+), XGBoost 3.1+ per-class base_score
- **LightGBM** — Text model files
- **scikit-learn** — Pickle files with 9 model types: GradientBoosting, RandomForest, IsolationForest, ExtraTrees, SVM (SVC/SVR/OneClass), Naive Bayes (Gaussian/Multinomial), GaussianProcessRegressor, KNN (Classifier/Regressor), Linear models, Pipelines
- **CatBoost** — JSON exports (oblivious trees)
- **ONNX** — TreeEnsemble, LinearClassifier/Regressor, SVMClassifier/Regressor, Normalizer, Scaler
- **URDF** — Robot description files → forward kinematics; outputs 4×4 homogeneous transform; inputs are joint angles

### Optimization Passes
1. Dead leaf elimination
2. Constant feature detection
3. Threshold quantization
4. Frequency-ordered branch sorting
5. Pipeline fusion (scaler → tree)
6. Vectorization analysis

### Output Targets
| Target | Description | Use Case |
|--------|-------------|----------|
| **C99** | Shared (`.so`) or static (`.a`) libraries | General deployment |
| **LLVM IR** | `.ll` text IR with configurable target triple | Hardware-specific optimization |
| **WebAssembly** | `.wat` + JS bindings | Browser and edge deployment |
| **MISRA-C:2012** | Safety-critical C with 8-rule compliance checker | Certified systems |
| **SIMD** | AVX2, AVX-512, ARM NEON/SVE, RISC-V V | Vectorized inference (2-8× speedup) |
| **GPU** | CUDA (sm75/sm86), Metal (Apple Silicon), OpenCL | Batch inference, edge GPUs |
| **FPGA HLS** | Xilinx Vitis, Intel SDK | Deterministic hardware acceleration |
| **Embedded** | Cortex-M4/M7, ESP32, STM32H7 | Microcontrollers, no-heap, static buffers |

### Production Features
- Deterministic JSON **audit trails** for regulatory compliance
- **Differential privacy** inference — Laplace and Gaussian noise mechanisms
- **Differential compilation** for incremental model updates
- **Ensemble composition** (voting, stacking)
- Multi-worker FastAPI server with `GET /api/metrics` (P50/P95/P99/P999 rolling window)
- Thread-safe, zero-allocation generated code
- **WCET analysis** — Worst-case execution time for Cortex-M4/M7, x86_64, AArch64, RISC-V
- **Safety certification** — DO-178C (aviation), ISO 26262 (automotive), IEC 62304 (medical) reports
- **Supply chain security** — Ed25519 signing, AES-256-GCM encryption, TPM 2.0 hooks
- **Deployment bundles** — Air-gapped tar.gz with manifests and signatures
- **Native servers** — C++ gRPC/HTTP, ROS 2 nodes, PX4 module skeletons
- 650+ test nuclear-grade test suite (core + acceleration + safety)

## Next Steps

→ [Getting Started](/docs/getting-started) — load and serve your first model in 60 seconds
