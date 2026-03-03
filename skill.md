# Timber — Coding Agent Skill Reference

> **For AI coding agents.** This file teaches you everything needed to use,
> integrate, and extend Timber without reading any other file.
> Version: 0.2.0

---

## What Timber does

Timber compiles a trained tree-based ML model (XGBoost, LightGBM, scikit-learn,
CatBoost, ONNX) into a self-contained C99 shared library and serves it over an
Ollama-compatible HTTP API. The compiled artifact has zero runtime dependencies
and runs at ~2 µs per sample.

**When to reach for Timber:**
- You need sub-millisecond tree model inference without Python overhead
- You want to serve a classical ML model the same way Ollama serves LLMs
- You need a portable C artifact for embedded, edge, or regulated environments
- You want a drop-in Ollama-compatible REST endpoint for a tree model

---

## Installation

```bash
pip install timber-compiler          # core (XGBoost, LightGBM, sklearn, ONNX, CatBoost)
pip install "timber-compiler[serve]" # + uvicorn for production HTTP
pip install "timber-compiler[full]"  # + all optional deps
```

**System requirement:** `gcc` or `clang` must be on `PATH`.

```bash
# verify
gcc --version || clang --version
python -c "import timber; print(timber.__version__)"
```

---

## The One-Liner (fastest path)

```bash
# Serve any model from a URL — no pre-download, no load step
timber serve https://yourhost.com/model.json
```

Timber downloads, parses, optimizes, compiles, and starts serving in one shot.
The model is cached at `~/.timber/` so subsequent `timber serve <name>` starts instantly.

---

## CLI Reference

### timber serve

```bash
# Serve from URL (downloads, compiles, serves)
timber serve https://example.com/model.json

# Serve from URL with explicit name
timber serve https://example.com/model.json --name my-model

# Serve a model already in the store
timber serve my-model

# Options
timber serve my-model --port 8080        # default: 11434
timber serve my-model --host 0.0.0.0     # default: 127.0.0.1
timber serve my-model --host 127.0.0.1   # localhost only (safe default)
```

### timber load

```bash
# Load a local model file into the store (auto-detects format)
timber load model.json --name my-model

# Explicit format
timber load model.json   --name my-model --format xgboost
timber load model.txt    --name lgb-model --format lightgbm
timber load model.pkl    --name rf-model  --format sklearn
timber load model.onnx   --name onnx-model --format onnx
timber load model.json   --name cat-model --format catboost
```

### timber pull

```bash
# Download + compile from URL, store in registry (does NOT serve)
timber pull https://example.com/model.json --name my-model

# Force re-download even if cached
timber pull https://example.com/model.json --name my-model --force
```

### timber list

```bash
timber list              # list all models in local store
```

Output columns: `NAME  FORMAT  TREES  FEATURES  OBJECTIVE  SIZE  COMPILED`

### timber inspect

```bash
timber inspect my-model  # show model metadata, tree stats, feature names
```

### timber remove

```bash
timber remove my-model   # delete model from store
```

### timber compile

```bash
# Compile to a standalone artifact directory (not stored in registry)
timber compile model.json --out ./dist/

# With explicit format
timber compile model.json --format xgboost --out ./dist/
```

Output files in `./dist/`:
- `model.c` — inference logic
- `model.h` — public C API header
- `model_data.c` — static tree data arrays
- `model.timber.json` — serialized IR (for inspection)
- `libtimber_model.so` — compiled shared library (Linux) / `.dylib` (macOS)
- `audit_report.json` — compilation audit trail with SHA-256 hashes

### timber inspect

```bash
timber inspect my-model
```

### timber validate

```bash
# Check compiled predictions match source framework within tolerance
timber validate --artifact ./dist/ --reference model.json --tolerance 1e-4
```

### timber bench

```bash
timber bench my-model --n 10000   # benchmark inference latency
```

---

## HTTP API

Default base URL: `http://localhost:11434`

### POST /api/predict

```bash
curl http://localhost:11434/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "inputs": [[1.2, 0.4, 3.1, 0.9, 2.2]]
  }'
```

**Request schema:**
```json
{
  "model": "<name>",
  "inputs": [[f1, f2, ..., fN], [f1, f2, ..., fN]]
}
```

**Response schema:**
```json
{
  "model": "my-model",
  "outputs": [[0.031]],
  "n_samples": 1,
  "latency_us": 1.8
}
```

**Output shape by objective:**

| Objective | outputs shape | Value |
|-----------|--------------|-------|
| `reg:squarederror` | `[n, 1]` | raw prediction |
| `reg:logistic` | `[n, 1]` | sigmoid(raw) ∈ (0,1) |
| `binary:logistic` | `[n, 1]` | P(positive class) ∈ (0,1) |
| `multi:softprob` | `[n, K]` | per-class probabilities, sum=1 |
| `multi:softmax` | `[n, K]` | per-class probabilities, sum=1 |
| `rank:pairwise` | `[n, 1]` | raw ranking score |

### POST /api/generate (Ollama-compatible alias)

Identical to `/api/predict`. Use this for Ollama drop-in compatibility.

### GET /api/models

```bash
curl http://localhost:11434/api/models
```

```json
{
  "models": [
    {
      "name": "my-model",
      "framework": "xgboost",
      "n_trees": 50,
      "n_features": 30,
      "objective": "binary:logistic",
      "size_bytes": 49152
    }
  ]
}
```

### GET /api/health

```bash
curl http://localhost:11434/api/health
```

```json
{"status": "ok", "version": "0.2.0"}
```

---

## Python API

### High-level: parse + optimize + emit in one call

```python
from timber.frontends.auto_detect import parse_model
from timber.optimizer.pipeline import OptimizerPipeline
from timber.codegen.c99 import C99Emitter

# 1. Parse any supported format
ir = parse_model("model.json")          # auto-detects format
ir = parse_model("model.json", fmt="xgboost")  # explicit

# 2. Optimize (6 passes, all enabled by default)
optimizer = OptimizerPipeline()
result = optimizer.run(ir)
optimized_ir = result.ir

# 3. Emit C99 source
emitter = C99Emitter()
output = emitter.emit(optimized_ir)

# 4. Write to disk
files = output.write("./dist/")
# files is a dict: {"model.c": Path, "model.h": Path, "model_data.c": Path}
```

### TimberPredictor (in-process inference, no HTTP)

```python
from timber.runtime.predictor import TimberPredictor
import numpy as np

predictor = TimberPredictor.from_store("my-model")   # from registry
predictor = TimberPredictor.from_artifact("./dist/")  # from directory

# Single sample
x = np.array([[1.2, 0.4, 3.1, 0.9]], dtype=np.float32)
y = predictor.predict(x)       # shape: (1, n_outputs)

# Batch
X = np.random.rand(1000, 30).astype(np.float32)
Y = predictor.predict(X)       # shape: (1000, n_outputs)
```

### Model store operations

```python
from timber.store import ModelStore

store = ModelStore()                       # default ~/.timber/
store = ModelStore("/custom/path")         # custom location

# List
models = store.list()                      # list[ModelInfo]
for m in models:
    print(m.name, m.n_trees, m.objective)

# Load from file
store.load("model.json", name="my-model")

# Pull from URL
store.pull("https://example.com/model.json", name="my-model")

# Remove
store.remove("my-model")

# Get artifact path
path = store.artifact_path("my-model")    # Path to artifact directory
```

### Framework-specific parsers (direct use)

```python
# XGBoost
from timber.frontends.xgboost_parser import parse_xgboost_model
ir = parse_xgboost_model("model.json")

# LightGBM
from timber.frontends.lightgbm_parser import parse_lightgbm_model
ir = parse_lightgbm_model("model.txt")

# scikit-learn
from timber.frontends.sklearn_parser import parse_sklearn_model
ir = parse_sklearn_model("model.pkl")

# CatBoost
from timber.frontends.catboost_parser import parse_catboost_model
ir = parse_catboost_model("model.json")

# ONNX
from timber.frontends.onnx_parser import parse_onnx_model
ir = parse_onnx_model("model.onnx")
```

### TimberIR inspection

```python
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")
ensemble = ir.pipeline[-1]             # TreeEnsembleStage (always last)

print(ensemble.n_trees)                # int
print(ensemble.n_features)            # int
print(ensemble.n_classes)             # 1 (regression/binary) or K (multiclass)
print(ensemble.objective)             # Objective enum
print(ensemble.base_score)            # float
print(ensemble.per_class_base_scores) # list[float] — multiclass only
print(ensemble.learning_rate)         # float
print(ensemble.is_boosted)            # True = GBT, False = RF

for tree in ensemble.trees:
    print(tree.tree_id, tree.n_nodes, tree.max_depth)

print(ir.schema.fields)               # list[Field] — feature names + types
print(ir.metadata.source_framework)   # "xgboost" | "lightgbm" | ...
```

### WebAssembly backend

```python
from timber.codegen.wasm import WasmEmitter
from timber.frontends.auto_detect import parse_model
from timber.optimizer.pipeline import OptimizerPipeline

ir = parse_model("model.json")
result = OptimizerPipeline().run(ir)
files = WasmEmitter().emit(result.ir).write("./wasm-dist/")
# files["model.wat"]       — WebAssembly Text Format
# files["timber_model.js"] — JavaScript bindings
```

---

## Complete End-to-End Workflows

### Workflow 1: Train → Load → Serve → Predict (XGBoost)

```python
# Step 1: Train and save
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = xgb.XGBClassifier(n_estimators=50, max_depth=4)
model.fit(X_train, y_train)
model.get_booster().save_model("model.json")
```

```bash
# Step 2: Load and serve
timber load model.json --name bc-model
timber serve bc-model
```

```bash
# Step 3: Predict
curl http://localhost:11434/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "bc-model", "inputs": [[17.99, 10.38, 122.8, ...]]}'
```

### Workflow 2: Serve from URL (zero setup)

```bash
pip install timber-compiler
timber serve https://example.com/fraud_model.json --name fraud
```

```python
# Python client
import requests, numpy as np

resp = requests.post("http://localhost:11434/api/predict", json={
    "model": "fraud",
    "inputs": np.random.rand(1, 30).tolist()
})
print(resp.json()["outputs"])  # [[0.031]]
```

### Workflow 3: Compile to standalone C (no Timber at runtime)

```bash
timber compile model.json --format xgboost --out ./dist/
```

```bash
# Build shared library anywhere (no Python needed)
gcc -O2 -shared -fPIC -o libtimber_model.so dist/model.c dist/model_data.c -lm

# Or a static binary for embedding
gcc -O2 -o my_app my_app.c dist/model.c dist/model_data.c -lm
```

```c
// my_app.c
#include "model.h"
#include <stdio.h>

int main() {
    float input[30] = {17.99f, 10.38f, /* ... */};
    float output[1];
    TimberCtx ctx;
    timber_init(&ctx);
    timber_infer_single(input, output, &ctx);
    printf("P(malignant) = %.4f\n", output[0]);
    return 0;
}
```

### Workflow 4: sklearn Pipeline with scaler

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(n_estimators=100))
])
pipe.fit(X_train, y_train)

with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)
```

```bash
# Timber detects the scaler, folds it into tree thresholds (pipeline fusion pass)
# Generated C has NO scaler code — pure tree inference
timber load pipeline.pkl --name pipe-model --format sklearn
timber serve pipe-model
```

### Workflow 5: Multiclass XGBoost

```python
import xgboost as xgb
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = xgb.XGBClassifier(n_estimators=50, objective="multi:softprob", num_class=3)
model.fit(X, y)
model.get_booster().save_model("iris.json")
```

```bash
timber load iris.json --name iris
timber serve iris
```

```bash
# outputs is shape [1, 3] — per-class probabilities
curl http://localhost:11434/api/predict \
  -d '{"model": "iris", "inputs": [[5.1, 3.5, 1.4, 0.2]]}'
# {"outputs": [[0.972, 0.015, 0.013]], "n_samples": 1}
```

---

## C API Reference (generated `model.h`)

```c
// Context struct — stack-allocate, never heap
typedef struct { int initialized; } TimberCtx;

// Initialize context (call once before inference)
void timber_init(TimberCtx *ctx);

// Single-sample inference
// inputs:  float array of length TIMBER_N_FEATURES
// outputs: float array of length TIMBER_N_OUTPUTS
void timber_infer_single(const float *inputs, float *outputs, TimberCtx *ctx);

// Batch inference
// inputs:  float[n_samples][TIMBER_N_FEATURES] (row-major)
// outputs: float[n_samples][TIMBER_N_OUTPUTS]  (row-major)
void timber_infer(const float *inputs, int n_samples, float *outputs, TimberCtx *ctx);

// Constants (values set at compile time from the model)
#define TIMBER_N_FEATURES  30
#define TIMBER_N_OUTPUTS   1    // K for multiclass
#define TIMBER_N_TREES     50
#define TIMBER_VERSION     "0.2.0"
```

---

## Error Reference

| Error message | Cause | Fix |
|--------------|-------|-----|
| `gcc: command not found` | No C compiler on PATH | `sudo apt install gcc` or `xcode-select --install` |
| `Format not recognized` | File extension ambiguous | Add `--format xgboost` (or lightgbm/sklearn/onnx/catboost) |
| `Model 'x' not found` | Name not in registry | Run `timber list` to see registered names |
| `expected N features, got M` | Input shape wrong | Check `timber inspect <name>` for feature count |
| `Connection refused :11434` | Server not running | Run `timber serve <name>` first |
| `HTTP 413` | Request body > 64 MB | Split into smaller batches |
| `HTTP 400` | Malformed JSON or wrong Content-Type | Add `-H "Content-Type: application/json"` |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TIMBER_HOME` | `~/.timber` | Root directory for model store |
| `TIMBER_CC` | `gcc` (→ `clang`) | C compiler to use for compilation |
| `TIMBER_LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `TIMBER_DEFAULT_PORT` | `11434` | HTTP server default port |

---

## Key Constraints & Gotchas

**Model format requirements:**
- XGBoost: must be JSON (`booster.save_model("model.json")`), not binary `.ubj`
- CatBoost: must be JSON export (`model.save_model("m.json", format="json")`), not native `.cbm`
- scikit-learn: standard pickle; supports `GradientBoostingClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `DecisionTreeClassifier`, and `Pipeline` wrapping any of these

**Input dtype:**
- Always pass `float32` arrays to `TimberPredictor.predict()` — it does NOT auto-cast
- HTTP API accepts any JSON number; internally cast to float32

**Output interpretation:**
- Binary classification: `outputs[0][0]` is `P(positive_class)` — threshold at 0.5 for class label
- Multiclass: `outputs[0]` is a list of K probabilities; `argmax` gives class index
- Regression: `outputs[0][0]` is the raw predicted value

**Concurrency:**
- `TimberCtx` is read-only after `timber_init()` — C inference is thread-safe
- HTTP server is single-threaded in v0.2.0 — run multiple instances behind a load balancer for concurrent workloads

**Portability:**
- `.so` / `.dylib` is NOT portable across architectures — recompile on target machine
- The C source files (`model.c`, `model_data.c`) ARE portable — copy and compile anywhere

**Name constraints:**
- Model names: `[a-z0-9_-]+` only (lowercase, digits, hyphens, underscores)
- Names are case-sensitive

---

## Optimizer Passes (what happens automatically)

When you run `timber load` / `timber serve` / `timber compile`, these 6 passes run:

1. **Dead Leaf Elimination** — removes nodes where both subtrees give identical output
2. **Constant Feature Detection** — folds splits on features that never change (requires calibration data)
3. **Threshold Quantization** — downcasts float64 thresholds to float32 (reduces size, improves cache)
4. **Frequency Branch Sort** — reorders branches by frequency (improves CPU branch prediction; requires calibration data)
5. **Pipeline Fusion** — folds `StandardScaler` / `MinMaxScaler` into tree thresholds (removes scaler from generated C)
6. **Vectorization Analysis** — advisory pass for future SIMD backend

All passes are idempotent. Passes 2 and 4 are no-ops without calibration data.

---

## File Layout After `timber compile --out ./dist/`

```
dist/
├── model.c              # inference logic — traverse_tree(), timber_infer_single(), timber_infer()
├── model.h              # public API — extern "C" safe, include in C or C++
├── model_data.c         # static const arrays: features, thresholds, left/right/leaf/is_leaf per tree
├── model.timber.json    # serialized TimberIR — inspect the parsed+optimized model
├── libtimber_model.so   # compiled shared library (Linux) — loaded by TimberPredictor
└── audit_report.json    # SHA-256 hashes, pass timings, model summary — for compliance
```

**model_data.c structure (one set per tree):**
```c
static const int32_t tree_0_features[N]     = { ... };
static const float   tree_0_thresholds[N]   = { ... };
static const int32_t tree_0_left[N]         = { ... };
static const int32_t tree_0_right[N]        = { ... };
static const float   tree_0_leaves[N]       = { ... };
static const int8_t  tree_0_is_leaf[N]      = { ... };
static const int8_t  tree_0_default_left[N] = { ... };  // NaN routing
static const float   TIMBER_BASE_SCORE      = 0.5f;
```

---

## Adding a New Frontend (for contributors)

```python
# 1. timber/frontends/my_framework_parser.py
from timber.ir.model import TimberIR, TreeEnsembleStage, Tree, TreeNode, Objective

def parse_my_framework_model(path: str) -> TimberIR:
    # ... read file, build TreeNode/Tree/TreeEnsembleStage objects ...
    return TimberIR(schema=..., pipeline=[ensemble], metadata=...)

# 2. Register in timber/frontends/auto_detect.py
# 3. Add tests in tests/
```

---

## Quick Decision Guide

| Goal | Command / API |
|------|--------------|
| Serve model from URL immediately | `timber serve <url>` |
| Load local model, serve by name | `timber load f.json --name x` then `timber serve x` |
| Compile to standalone C only | `timber compile f.json --out ./dist/` |
| In-process Python inference (no HTTP) | `TimberPredictor.from_store("x").predict(X)` |
| Check prediction accuracy vs. source | `timber validate --artifact ./dist/ --reference f.json` |
| Benchmark latency | `timber bench x --n 10000` |
| Inspect model (trees, features, objective) | `timber inspect x` |
| List all local models | `timber list` |
| Delete a model | `timber remove x` |
| Embed in C/C++ without Python | Compile `model.c` + `model_data.c`, include `model.h` |
| Deploy to browser / edge (WASM) | `WasmEmitter().emit(ir).write("./wasm/")` |
