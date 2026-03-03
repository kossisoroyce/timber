# Architecture

This page describes how Timber works internally — from loading a model file to
serving a compiled binary over HTTP.

---

## Overview

Timber is a **multi-stage compiler** for tree-based ML models. It has the same
conceptual structure as a traditional compiler (frontend → IR → optimizer →
backend) but specializes in the tree ensemble domain.

```
  Source model file
        │
        ▼
  ┌─────────────┐
  │   Frontend  │  Parse framework-native format → TimberIR
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  Optimizer  │  6 IR-level passes (dead leaves, fusion, quantization, …)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │   Backend   │  Emit C99 / WASM / MISRA-C source
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │   Compile   │  gcc / clang → .so / .dylib
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │    Serve    │  ctypes loader + HTTP server
  └─────────────┘
```

---

## Stage 1 — Frontend (Parse)

**Code:** `timber/frontends/`

Each supported framework has a dedicated parser that converts the native model
format into a **TimberIR** object — a framework-agnostic typed AST.

| Parser | Input | Key challenge |
|--------|-------|---------------|
| `xgboost_parser.py` | XGBoost JSON | Per-class base_score vectors (3.1+), learning_rate location |
| `lightgbm_parser.py` | LightGBM text | Custom tree serialization format |
| `sklearn_parser.py` | pickle | Dynamic type dispatch across estimator types |
| `onnx_parser.py` | ONNX protobuf | ML opset operator mapping |
| `catboost_parser.py` | CatBoost JSON | Oblivious tree structure |

`auto_detect.py` selects the right parser from file extension and content
inspection (e.g., presence of `"learner"` key for XGBoost, `"oblivious_trees"`
for CatBoost).

---

## Stage 2 — Timber IR

**Code:** `timber/ir/model.py`

The IR is a set of Python `@dataclass` objects. The top level is `TimberIR`:

```
TimberIR
├── schema: Schema                  # input feature definitions
│   └── fields: list[Field]
├── pipeline: list[PipelineStage]   # ordered transform stages
│   ├── ScalerStage?                # optional preprocessing
│   └── TreeEnsembleStage           # the tree ensemble
└── metadata: Metadata              # provenance, hashes, framework version
```

### Why a typed IR?

The IR provides a **stable interface** between the framework-specific frontends
and the backends. This means:

- Adding a new framework (frontend) does not touch the optimizer or backends
- Adding a new target (backend) does not touch the parsers
- The optimizer works on a single well-defined data structure

The IR is also **serializable to JSON** (`model.timber.json`) for inspection and
debugging.

### Tree representation

Trees are stored as flat arrays of `TreeNode` objects (not pointer-linked nodes).
Each node stores: `feature_index`, `threshold`, `left_child`, `right_child`,
`leaf_value`, `is_leaf`, `default_left` (NaN routing), `depth`.

This flat representation maps directly to the static const arrays in the
generated `model_data.c`, avoiding any heap allocation at runtime.

---

## Stage 3 — Optimizer

**Code:** `timber/optimizer/pipeline.py`

The optimizer runs up to 6 passes over the IR. Each pass is idempotent and
reports whether it changed the IR.

### Pass 1: Dead Leaf Elimination

Traverses each tree bottom-up. If an internal node's left and right subtrees
both produce identical values for all possible inputs, the node is replaced
by a single leaf with that value. This is the most impactful pass for shallow
trees trained on noisy data.

### Pass 2: Constant Feature Detection

If a feature has the same value across all calibration samples (optional), all
split nodes on that feature are replaced with a direct branch to the
deterministic child. Requires `--calibration-data`.

### Pass 3: Threshold Quantization

Converts `float64` split thresholds to `float32`. If the rounding changes any
split decision for any sample in calibration data, the threshold is kept as
`float64`. Otherwise it is downcast, reducing `model_data.c` size and improving
cache locality.

### Pass 4: Frequency Branch Sort

Reorders left/right children so the more-frequently-taken path is always
"left". In the generated C, the `if (val < threshold)` branch falls through
naturally on modern CPUs. Requires `--calibration-data`.

### Pass 5: Pipeline Fusion

When a `ScalerStage` (e.g., `StandardScaler`) immediately precedes a
`TreeEnsembleStage`, the scaler is **folded into the tree thresholds**:

```
threshold_new[i] = (threshold_old[i] - mean[feature]) / scale[feature]
```

The `ScalerStage` is removed from the pipeline entirely. The generated C
has no scaler computation at all.

### Pass 6: Vectorization Analysis

Analyzes whether SIMD vectorization applies across trees (advisory). Does not
yet emit intrinsics — this is preparation for a future LLVM IR backend.

---

## Stage 4 — C99 Backend

**Code:** `timber/codegen/c99.py`

The `C99Emitter` takes a `TimberIR` + `TargetSpec` and produces three files:

### model_data.c

Static const arrays holding all tree node data. One set of 7 arrays per tree:

```c
static const int32_t tree_0_features[N]    = { ... };
static const float   tree_0_thresholds[N]  = { ... };
static const int32_t tree_0_left[N]        = { ... };
static const int32_t tree_0_right[N]       = { ... };
static const float   tree_0_leaves[N]      = { ... };
static const int8_t  tree_0_is_leaf[N]     = { ... };
static const int8_t  tree_0_default_left[N]= { ... };
```

Plus base scores:
```c
static const float  TIMBER_BASE_SCORE              = 0.5f;
static const double TIMBER_CLASS_BASE_SCORES[K]    = { ... }; /* multiclass only */
```

### model.c

Contains the inference logic:

1. **`traverse_tree()`** — iterative (non-recursive) tree traversal. Bounded
   by `max_depth + 2` iterations. Handles NaN via `default_left` flags.

2. **`timber_infer_single()`** — unrolled calls to `traverse_tree()` for every
   tree. For multiclass, accumulates per-class sums then applies softmax in
   double precision. For binary, applies sigmoid. For regression, returns raw sum.

3. **`timber_infer()`** — batch loop over `timber_infer_single()`.

### Key design decisions

- **No heap allocation** — all arrays are `static const`. `TimberCtx` is a
  stack-allocated struct with a single `int initialized` field.
- **No recursion** — `traverse_tree()` is a bounded while loop. Safe for
  embedded targets with limited stack.
- **Double-precision accumulation** — raw scores accumulate in `double` before
  conversion to the output float type. Matches framework precision.
- **Numerically stable softmax** — max subtraction before exp to avoid overflow.

---

## Stage 5 — Compile

**Code:** `timber/store.py` (load_model), `timber/runtime/predictor.py`

After emitting C source, Timber invokes:

```bash
gcc -O2 -shared -fPIC -o libtimber_model.so model.c -lm
```

The compiler (`gcc` → `clang` fallback) must be on `PATH`. The resulting shared
library is loaded via Python `ctypes`:

```python
lib = ctypes.CDLL(lib_path)
lib.timber_infer.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # inputs
    ctypes.c_int,                    # n_samples
    ctypes.POINTER(ctypes.c_float),  # outputs
    ctypes.c_void_p,                 # ctx
]
```

`numpy` arrays are passed as raw C pointers using `.ctypes.data_as()`.
Temporary directories created during compilation are cleaned up via `atexit`.

---

## Stage 6 — HTTP Server

**Code:** `timber/serve.py`

The server is built on Python's stdlib `http.server.BaseHTTPRequestHandler`.
It has no third-party dependencies in the default install.

Key implementation details:

- **Body size limit** — `Content-Length` is checked before reading; requests
  exceeding 64 MB return HTTP 413.
- **Content-Length errors** — `ValueError` during header parse returns HTTP 400.
- **JSON routing** — `/api/predict` and `/api/generate` share the same handler.
- **Concurrent safety** — `TimberCtx` is read-only after `timber_init()`, so
  the C inference calls are safe from multiple threads. The Python HTTP layer
  is single-threaded in the current version.

---

## Model Store

**Code:** `timber/store.py`

The store manages the `~/.timber/` directory tree.

**Atomic writes** — `registry.json` is never written directly. Updates go to
`.json.tmp` then renamed with `os.replace()`, which is atomic on POSIX. This
prevents registry corruption on crash.

**Name sanitization** — model names are validated against `[a-z0-9_-]+` before
any filesystem operation. This prevents path traversal attacks.

**Cache** — URL downloads are cached at `~/.timber/cache/<sha256(url)[:16]>/`.
Re-use is automatic on repeated `timber pull` of the same URL. `--force`
bypasses the cache.

---

## WebAssembly Backend

**Code:** `timber/codegen/wasm.py`

Emits WebAssembly Text Format (`.wat`) for browser and edge deployment. The
same flat-array tree representation used in C maps directly to WAT linear
memory. JavaScript bindings (`timber_model.js`) wrap the WASM module.

---

## MISRA-C Backend (Roadmap)

**Code:** `timber/codegen/misra_c.py`

Targets ISO 26262 (automotive) and IEC 62304 (medical) certification workflows.
Full MISRA-C:2012 compliance checking is on the roadmap.

---

## Audit Trail

Every compilation writes `audit_report.json` to the artifact directory:

```json
{
  "timber_version": "0.2.0",
  "timestamp": "2026-03-03T12:00:00Z",
  "input_hash": "sha256:abc123...",
  "model_summary": {
    "n_trees": 50,
    "n_features": 30,
    "objective": "binary:logistic"
  },
  "passes": [
    {"name": "dead_leaf_elimination", "changed": true,  "duration_ms": 1.2},
    {"name": "pipeline_fusion",       "changed": true,  "duration_ms": 0.4},
    {"name": "threshold_quantization","changed": false, "duration_ms": 0.8}
  ],
  "output_files": {
    "model.c":      "sha256:def456...",
    "model_data.c": "sha256:ghi789..."
  }
}
```

This supports compliance requirements in regulated industries (SOX, MiFID II,
FDA 21 CFR Part 11, IEC 62304).
