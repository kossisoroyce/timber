---
sidebar_position: 1
slug: /
title: Introduction
---

# Timber

**Ollama for classical ML models.** Compile and serve tree-based models at native speed.

Timber is an ahead-of-time (AOT) compiler that transforms trained machine learning models — XGBoost, LightGBM, scikit-learn, CatBoost, and ONNX — into optimized, self-contained C99 inference code. No Python runtime at inference time. No dynamic allocation. No recursion. Just fast, auditable, portable code.

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
- **XGBoost** — JSON model dumps (v2.0+)
- **LightGBM** — Text model files
- **scikit-learn** — Pickle files (GradientBoosting, RandomForest, Pipeline)
- **CatBoost** — JSON exports (oblivious trees)
- **ONNX** — TreeEnsemble operators from ML opset

### Optimization Passes
1. Dead leaf elimination
2. Constant feature detection
3. Threshold quantization
4. Frequency-ordered branch sorting
5. Pipeline fusion (scaler → tree)
6. Vectorization analysis

### Output Targets
- **C99** — Shared libraries for servers, embedded systems
- **WebAssembly** — Browser and edge deployment
- **MISRA-C** — Safety-critical (automotive, medical, avionics)

### Production Features
- Deterministic JSON **audit trails** for regulatory compliance
- **Differential compilation** for incremental model updates
- **Ensemble composition** (voting, stacking)
- Thread-safe, zero-allocation generated code

## Next Steps

→ [Getting Started](/docs/getting-started) — load and serve your first model in 60 seconds
