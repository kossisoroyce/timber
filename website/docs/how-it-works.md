---
sidebar_position: 3
title: How It Works
---

# How It Works

Timber is a classical compiler that treats trained ML models as program specifications.

## The Compiler Pipeline

```
Model Artifact (.json, .pkl, .txt, .onnx)
    │
    ▼
┌──────────────────┐
│   Front-End      │  Format-specific parsers
│   (5 parsers)    │  → Framework-agnostic IR
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Optimizer      │  6 domain-specific passes
│   (6 passes)     │  → Optimized IR
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Back-End       │  3 code emitters
│   (C99/WASM/     │  → Self-contained source
│    MISRA-C)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   C Compiler     │  gcc -O3 -shared
│   (gcc/clang)    │  → .so / .dylib
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Model Store    │  ~/.timber/models/
│   + HTTP Server  │  → REST API on :11434
└──────────────────┘
```

## Phase 1: Front-End (Parsing)

Each supported framework has a dedicated parser that converts its native format into Timber's Intermediate Representation (IR).

| Framework | Parser | Input Format | Key Details |
|-----------|--------|-------------|-------------|
| XGBoost | `xgboost_parser` | JSON dump | Converts `base_score` from probability to logit space |
| LightGBM | `lightgbm_parser` | Text model | Handles negative-indexed leaf references |
| scikit-learn | `sklearn_parser` | Pickle | Supports Pipelines with StandardScaler |
| CatBoost | `catboost_parser` | JSON export | Expands oblivious (symmetric) trees |
| ONNX | `onnx_parser` | Protobuf | Handles TreeEnsemble ML operators |

**Auto-detection** inspects file extension and content to select the right parser automatically.

## Phase 2: Optimization

The optimizer runs 6 passes sequentially, each transforming the IR:

### Pass 1: Dead Leaf Elimination
Prunes leaves whose contribution is negligible relative to the maximum leaf value. When both children of a node are pruned, the node collapses to a leaf. **Effect:** Reduces tree depth and code size.

### Pass 2: Constant Feature Detection
Folds internal nodes where both children have identical leaf values — the split is redundant. **Effect:** Eliminates unnecessary comparisons.

### Pass 3: Threshold Quantization
Analyzes all split thresholds per feature to determine minimum precision (int8, int16, float16, float32). Stores precision metadata for potential SIMD backends. **Effect:** Enables future narrower-type optimizations.

### Pass 4: Frequency-Ordered Branch Sorting
Given calibration data, counts branch frequencies and reorders children so the most-taken branch is the fall-through path. **Effect:** Better branch prediction and I-cache utilization.

### Pass 5: Pipeline Fusion
Absorbs a preceding `ScalerStage` into tree thresholds: `θ' = θ × σ + μ`. Eliminates the entire preprocessing step. **Effect:** Zero-cost feature scaling.

### Pass 6: Vectorization Analysis
Analyzes tree structure to identify SIMD batching opportunities — depth profiles, feature access patterns, structurally identical tree groups. Produces `VectorizationHint` annotations. **Effect:** Guides future SIMD code generation.

Each pass produces an **audit log** documenting what changed and timing.

## Phase 3: Code Generation

The C99 emitter produces five files:

| File | Contents |
|------|----------|
| `model.h` | Public API, constants (`TIMBER_N_FEATURES`, `TIMBER_N_OUTPUTS`), ABI version |
| `model_data.c` | All tree data as `static const` arrays (thresholds, feature indices, children, leaf values) |
| `model.c` | Inference logic — iterative tree traversal, accumulation, activation function |
| `CMakeLists.txt` | CMake build configuration |
| `Makefile` | GNU Make fallback |

### Design Guarantees

The generated code is designed for the most constrained environments:

- **No `malloc`** — all data is compile-time constant
- **No recursion** — tree traversal is iterative with bounded loop count
- **No library dependencies** — only `<math.h>` for `exp()` in sigmoid/softmax
- **Double-precision accumulation** — sums tree outputs in `double` before final `float` cast
- **NaN handling** — missing values follow `default_left` path per XGBoost/LightGBM semantics
- **Thread-safe** — context is read-only after init; concurrent inference is safe
- **ABI versioned** — `TIMBER_ABI_VERSION` for compatibility detection

## Phase 4: Serving

The compiled shared library is loaded via Python `ctypes`. The HTTP server (port 11434) handles:
- JSON parsing and response serialization (Python)
- Buffer allocation and copying (Python)
- **Actual inference** (compiled C, via ctypes)

This architecture means Python is never in the inference hot path. The C function call itself takes ~2 µs; the HTTP overhead adds ~89 µs for a total of ~91 µs per request.
