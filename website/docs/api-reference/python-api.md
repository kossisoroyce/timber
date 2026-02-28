---
sidebar_position: 3
title: Python API
---

# Python API Reference

## `timber.store.ModelStore`

Manages the local model store at `~/.timber/models/`.

### Constructor

```python
from timber.store import ModelStore

store = ModelStore()                        # Default: ~/.timber
store = ModelStore(home=Path("/opt/timber"))  # Custom location
```

### Methods

#### `load_model(path, name=None, format=None) → dict`

Compile and cache a model.

```python
info = store.load_model("model.json", name="my-model")
# Returns: {"name": "my-model", "n_trees": 50, "n_features": 30, ...}
```

#### `list_models() → list[dict]`

List all cached models.

```python
for m in store.list_models():
    print(f"{m['name']}: {m['n_trees']} trees")
```

#### `get_model(name) → dict`

Get metadata for a specific model.

```python
info = store.get_model("my-model")
```

#### `remove_model(name) → bool`

Remove a cached model.

```python
store.remove_model("my-model")
```

#### `get_model_dir(name) → Path`

Get the filesystem path to a model's cache directory.

```python
model_dir = store.get_model_dir("my-model")
# ~/.timber/models/my-model/
```

#### `get_lib_path(name) → Path`

Get the path to the compiled shared library.

```python
lib_path = store.get_lib_path("my-model")
# ~/.timber/models/my-model/libtimber_model.so
```

---

## `timber.runtime.predictor.TimberPredictor`

Native inference from Python via `ctypes`.

### Constructor

```python
from timber.runtime.predictor import TimberPredictor

pred = TimberPredictor(
    lib_path="/path/to/libtimber_model.so",
    n_features=30,
    n_outputs=1,
    n_trees=50,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `lib_path` | `str \| Path` | Path to compiled `.so` / `.dylib` |
| `n_features` | `int` | Number of input features |
| `n_outputs` | `int` | Number of outputs per sample |
| `n_trees` | `int` | Number of trees |

### Methods

#### `predict(X) → np.ndarray`

Run inference on a NumPy array.

```python
import numpy as np

X = np.random.randn(100, 30).astype(np.float32)
outputs = pred.predict(X)
# outputs.shape == (100,) for single-output models
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_features` | `int` | Input feature count |
| `n_outputs` | `int` | Output count per sample |
| `n_trees` | `int` | Tree count |

---

## `timber.frontends.auto_detect`

### `detect_format(path) → str`

Auto-detect model format from file.

```python
from timber.frontends.auto_detect import detect_format

fmt = detect_format("model.json")  # "xgboost"
```

### `parse_model(path, format=None) → TimberIR`

Parse a model file into the Timber IR.

```python
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")
print(f"Trees: {len(ir.stages)}")
```

---

## `timber.optimizer.pipeline`

### `run(ir, calibration_data=None) → TimberIR`

Run all 6 optimization passes on an IR.

```python
from timber.optimizer.pipeline import run

optimized_ir = run(ir)
```

---

## `timber.codegen.c99.C99Emitter`

### `emit() → dict[str, str]`

Generate C99 source files.

```python
from timber.codegen.c99 import C99Emitter

emitter = C99Emitter(ir)
files = emitter.emit()
# files = {"model.h": "...", "model.c": "...", "model_data.c": "...", ...}
```
