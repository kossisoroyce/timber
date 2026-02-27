# Getting Started

Timber compiles trained tree-based ML models into native C code and serves them over HTTP — like Ollama, but for classical ML.

## Installation

```bash
pip install timber-compiler
```

**Requirements:** Python 3.10+ and a C compiler (`gcc` or `clang`) for compiling shared libraries.

## Your First Model in 60 Seconds

### 1. Train a model (or use an existing one)

```python
# train_model.py
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X_train, y_train)
model.get_booster().save_model("my_model.json")
print(f"Saved model with {model.n_estimators} trees, {X_train.shape[1]} features")
```

### 2. Load it into Timber

```bash
timber load my_model.json --name breast-cancer
```

Output:
```
Timber v0.1.0 — loading model...

Model loaded successfully:
  Name:      breast-cancer
  Format:    xgboost
  Framework: xgboost
  Trees:     50
  Features:  30
  Objective: binary:logistic
  Compiled:  yes
  Size:      58.0 KB

Run inference with:
  timber serve breast-cancer
```

### 3. Serve it

```bash
timber serve breast-cancer
```

### 4. Query it

```bash
curl http://localhost:11434/api/predict \
  -d '{"model": "breast-cancer", "inputs": [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]}'
```

Response:
```json
{
  "model": "breast-cancer",
  "outputs": [0.027],
  "n_samples": 1,
  "latency_us": 91.0,
  "done": true
}
```

## What Just Happened?

When you ran `timber load`, Timber:

1. **Auto-detected** the model format (XGBoost JSON)
2. **Parsed** the model into Timber's framework-agnostic IR
3. **Optimized** it through 6 compiler passes (dead leaf elimination, constant folding, threshold quantization, branch sorting, pipeline fusion, vectorization analysis)
4. **Emitted** optimized C99 source code
5. **Compiled** a native shared library (`.so` / `.dylib`)
6. **Cached** everything in `~/.timber/models/breast-cancer/`

When you ran `timber serve`, it loaded the pre-compiled library and started an HTTP server. The actual inference runs in compiled C — Python only handles the HTTP envelope.

## Managing Models

```bash
# List all loaded models
timber list

# Remove a model
timber remove breast-cancer
```

## Supported Formats

| Framework | File Types | Auto-Detected |
|-----------|------------|---------------|
| XGBoost | `.json` | ✅ |
| LightGBM | `.txt`, `.model` | ✅ |
| scikit-learn | `.pkl`, `.pickle` | ✅ |
| CatBoost | `.json` | ✅ |
| ONNX | `.onnx` | ✅ |

Use `--format` to override auto-detection:
```bash
timber load model.json --format catboost
```

## Next Steps

- [API Reference](api-reference.md) — full HTTP API and Python API docs
- [Examples](examples.md) — real-world usage for each framework
- [Advanced Usage](advanced.md) — direct compilation, C embedding, WASM output
- [Contributing](../CONTRIBUTING.md) — how to contribute to Timber
