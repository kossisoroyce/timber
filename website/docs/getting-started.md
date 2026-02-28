---
sidebar_position: 2
title: Getting Started
---

# Getting Started

Get a trained model running at native speed in 60 seconds.

## Prerequisites

- **Python 3.10+**
- **C compiler** (`gcc` or `clang`) — comes pre-installed on macOS and most Linux distros
- A trained tree-based model file (XGBoost, LightGBM, scikit-learn, CatBoost, or ONNX)

## Installation

```bash
pip install timber-compiler
```

Verify the install:

```bash
timber --help
```

## Quick Start

### 1. Load a Model

```bash
timber load model.json --name my-model
```

Timber will:
1. Auto-detect the framework (XGBoost, LightGBM, etc.)
2. Parse into a framework-agnostic IR
3. Run 6 optimization passes
4. Emit C99 source code
5. Compile a native shared library
6. Cache everything in `~/.timber/models/my-model/`

### 2. Serve It

```bash
timber serve my-model
```

The server starts on port `11434` (same as Ollama) and exposes a REST API.

### 3. Query It

```bash
curl http://localhost:11434/api/predict \
  -d '{"model": "my-model", "inputs": [[1.0, 2.0, 3.0]]}'
```

Response:

```json
{
  "model": "my-model",
  "outputs": [0.97],
  "n_samples": 1,
  "latency_us": 91.0,
  "done": true
}
```

### 4. Manage Models

```bash
# List all loaded models
timber list

# Remove a model
timber remove my-model
```

## What Happened Under the Hood?

When you ran `timber load`, the compiler pipeline executed:

```
model.json
  │
  ├─ Front-end: XGBoost JSON parser
  │    → Extracted 50 trees, 30 features, binary:logistic objective
  │    → Converted base_score from probability to logit space
  │
  ├─ Optimizer: 6 passes
  │    1. Dead leaf elimination (pruned near-zero leaves)
  │    2. Constant feature detection (folded redundant splits)
  │    3. Threshold quantization (analyzed precision requirements)
  │    4. Branch sorting (optimized for branch prediction)
  │    5. Pipeline fusion (absorbed scaler into thresholds)
  │    6. Vectorization analysis (computed SIMD hints)
  │
  ├─ Code generator: C99 emitter
  │    → model.h (public API with ABI version)
  │    → model_data.c (tree data as static const arrays)
  │    → model.c (inference logic — no malloc, no recursion)
  │    → CMakeLists.txt + Makefile
  │
  └─ Compiler: gcc -O3 -shared -std=c99
       → libtimber_model.so (48 KB)
```

When you ran `timber serve`, the HTTP server loaded the pre-compiled `.so` via Python `ctypes`. The actual inference call goes **directly to compiled C** — Python only handles the HTTP envelope (JSON parsing, buffer copying). That's why inference is 2 µs while the HTTP round-trip is ~91 µs.

## Don't Have a Model Yet?

Train a quick one:

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, _, y_train, _ = train_test_split(data.data, data.target, random_state=42)

model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X_train, y_train)
model.get_booster().save_model("model.json")
```

Then load and serve it:

```bash
timber load model.json --name breast-cancer
timber serve breast-cancer
```

## Next Steps

- **[How It Works](/docs/how-it-works)** — deep dive into the compiler pipeline
- **[Examples](/docs/examples/xgboost)** — per-framework walkthroughs
- **[API Reference](/docs/api-reference/cli)** — complete CLI and HTTP docs
