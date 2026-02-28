---
sidebar_position: 5
title: ONNX
---

# ONNX Example

Timber can compile any tree ensemble exported to ONNX format via the ML opset.

## Export a Model to ONNX

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

data = load_breast_cancer()
model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
model.fit(data.data, data.target)

initial_type = [("float_input", FloatTensorType([None, 30]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

## Load and Serve

```bash
timber load model.onnx --name onnx-bc
timber serve onnx-bc
```

## Supported ONNX Operators

Timber handles these ONNX ML opset operators:

- `TreeEnsembleClassifier`
- `TreeEnsembleRegressor`

The parser extracts flat node arrays (node IDs, feature IDs, thresholds, modes, tree IDs) and reconstructs the tree structures into the Timber IR.

## Why Use ONNX?

ONNX is framework-agnostic, so you can export from any compatible framework and compile with Timber. This is useful when:

- The original framework isn't directly supported by Timber
- You want a standardized export format in your pipeline
- You're already using ONNX for other models
