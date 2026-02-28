---
sidebar_position: 3
title: Python Predictor
---

# Python Predictor (No Server)

If you don't need HTTP, use `TimberPredictor` for direct native inference from Python.

## Basic Usage

```python
from timber.runtime.predictor import TimberPredictor
import numpy as np

# Compile and load in one step
pred = TimberPredictor.from_model("model.json")

# Single sample
X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
y = pred.predict(X)
print(f"Prediction: {y}")

# Batch inference
X_batch = np.random.randn(1000, 30).astype(np.float32)
y_batch = pred.predict(X_batch)
print(f"Predicted {len(y_batch)} samples")
```

## From Pre-Compiled Artifact

If you've already compiled with `timber load` or `timber compile`:

```python
pred = TimberPredictor(
    lib_path="~/.timber/models/my-model/libtimber_model.so",
    n_features=30,
    n_outputs=1,
    n_trees=50,
)
y = pred.predict(X)
```

## Properties

```python
pred.n_features   # Number of input features
pred.n_outputs    # Number of output values per sample
pred.n_trees      # Number of trees in ensemble
```

## Drop-In Replacement

`TimberPredictor.predict()` accepts and returns NumPy arrays, making it a drop-in replacement for framework `predict()` methods:

```python
# Before (XGBoost)
y = model.predict(X)

# After (Timber — 336× faster)
y = pred.predict(X)
```

## Thread Safety

The predictor is thread-safe. The underlying C context is read-only after initialization, so multiple threads can call `predict()` concurrently without locks.
