# API Reference

## CLI Commands

### `timber load <model_path>`

Compile and cache a model locally.

| Option | Default | Description |
|--------|---------|-------------|
| `--name NAME` | filename stem | Name to register the model as |
| `--format FMT` | auto-detect | Input format hint (`xgboost`, `lightgbm`, `sklearn`, `catboost`, `onnx`) |

```bash
timber load model.json
timber load model.json --name fraud-detector
timber load pipeline.pkl --format sklearn
```

### `timber serve <name>`

Serve a loaded model over HTTP.

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | `0.0.0.0` | Bind host |
| `--port PORT` | `11434` | Bind port |

```bash
timber serve my-model
timber serve my-model --port 8080
```

### `timber list`

List all loaded models in a table.

```bash
timber list
```

### `timber remove <name>`

Remove a cached model.

```bash
timber remove my-model
```

### `timber compile`

Compile a model to C99 source (without caching in the store).

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | required | Model artifact path |
| `--format FMT` | auto-detect | Input format hint |
| `--target PATH` | `x86_64_generic` | Target spec TOML file |
| `--out DIR` | `./dist` | Output directory |
| `--calibration-data PATH` | none | CSV for branch sorting pass |

```bash
timber compile --model model.json --out ./dist/
```

### `timber inspect <model_path>`

Print model summary without compiling.

### `timber validate`

Validate compiled output against reference predictions.

| Option | Description |
|--------|-------------|
| `--artifact DIR` | Compiled artifact directory |
| `--reference PATH` | Original model file |
| `--data PATH` | Validation CSV |
| `--tolerance FLOAT` | Max allowed error (default: 1e-5) |

### `timber bench`

Benchmark inference performance.

| Option | Description |
|--------|-------------|
| `--artifact DIR` | Compiled artifact directory |
| `--data PATH` | Input data CSV |
| `--batch-sizes LIST` | Comma-separated batch sizes |

---

## HTTP API

Default port: `11434` (same as Ollama).

### POST `/api/predict`

Run inference on a loaded model.

**Request:**
```json
{
  "model": "my-model",
  "inputs": [[1.0, 2.0, 3.0, ...]]
}
```

- `model` — name of the loaded model
- `inputs` — 2D array of shape `[n_samples, n_features]`

**Response:**
```json
{
  "model": "my-model",
  "outputs": [0.97],
  "n_samples": 1,
  "latency_us": 91.0,
  "done": true
}
```

**Errors:**
```json
{"error": "model 'xyz' not loaded"}
{"error": "expected 30 features, got 10"}
```

### POST `/api/generate`

Alias for `/api/predict` (Ollama compatibility).

### GET `/api/models`

List all loaded models.

**Response:**
```json
{
  "models": [
    {
      "name": "fraud-detector",
      "n_features": 30,
      "n_outputs": 1,
      "n_trees": 50,
      "objective": "binary:logistic",
      "framework": "xgboost",
      "format": "xgboost",
      "version": "0.1.0"
    }
  ]
}
```

### GET `/api/model/:name`

Get metadata for a specific model.

### GET `/api/health`

Health check.

**Response:**
```json
{"status": "ok", "version": "0.1.0"}
```

---

## Python API

### `TimberPredictor`

Drop-in replacement for framework `predict()` methods.

```python
from timber.runtime.predictor import TimberPredictor

# From a model file (compiles on-the-fly)
pred = TimberPredictor.from_model("model.json")
outputs = pred.predict(X)  # numpy array in, numpy array out

# From a pre-compiled artifact directory
pred = TimberPredictor.from_artifact("./dist/", build=True)
outputs = pred.predict(X)
```

**Properties:**
- `pred.n_features` — number of input features
- `pred.n_outputs` — number of output values per sample
- `pred.n_trees` — number of trees in the ensemble

### `ModelStore`

Programmatic access to the model store.

```python
from timber.store import ModelStore

store = ModelStore()

# Load a model
info = store.load_model("model.json", name="my-model")

# List models
for m in store.list_models():
    print(f"{m.name}: {m.n_trees} trees, {m.n_features} features")

# Get model info
info = store.get_model("my-model")

# Remove a model
store.remove_model("my-model")

# Get paths
model_dir = store.get_model_dir("my-model")
lib_path = store.get_lib_path("my-model")
```

---

## Generated C API

After compilation, the generated C code exposes:

```c
#include "model.h"

// Initialize context
TimberCtx* ctx;
int err = timber_init(&ctx);

// Single-sample inference
float inputs[TIMBER_N_FEATURES] = { /* ... */ };
float outputs[TIMBER_N_OUTPUTS];
timber_infer_single(inputs, outputs, ctx);

// Batch inference
float batch[256 * TIMBER_N_FEATURES];
float results[256 * TIMBER_N_OUTPUTS];
timber_infer(batch, 256, results, ctx);

// Error handling
if (err != TIMBER_OK) {
    printf("Error: %s\n", timber_strerror(err));
}

// Optional logging
void my_logger(int level, const char* msg) {
    printf("[timber][%d] %s\n", level, msg);
}
timber_set_log_callback(my_logger);

// Cleanup
timber_free(ctx);
```

**Error codes:**
| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `TIMBER_OK` | Success |
| -1 | `TIMBER_ERR_NULL` | Null pointer argument |
| -2 | `TIMBER_ERR_INIT` | Context not initialized |
| -3 | `TIMBER_ERR_BOUNDS` | Argument out of bounds |
