---
sidebar_position: 1
title: Loading Models
---

# Loading Models

`timber load` is the entry point for the compilation pipeline. It parses, optimizes, compiles, and caches a model in one step.

## Basic Usage

```bash
timber load model.json --name my-model
```

If `--name` is omitted, Timber uses the filename stem:

```bash
timber load fraud_detector.json
# Registered as "fraud_detector"
```

## Format Override

Timber auto-detects the format, but you can override:

```bash
timber load model.json --format catboost
timber load pipeline.pkl --format sklearn
```

## Supported Formats

| Extension | Default Detection | Framework |
|-----------|-------------------|-----------|
| `.json` (with `learner` key) | XGBoost | XGBoost ≥2.0 |
| `.json` (with `oblivious_trees` key) | CatBoost | CatBoost |
| `.txt`, `.model` | LightGBM | LightGBM |
| `.pkl`, `.pickle` | scikit-learn | scikit-learn |
| `.onnx` | ONNX | Any ONNX exporter |

## What Gets Cached

After loading, `~/.timber/models/<name>/` contains:

```
~/.timber/models/my-model/
├── model.c              # Inference logic
├── model.h              # Public API header
├── model_data.c         # Tree data (static const)
├── libtimber_model.so   # Compiled shared library
├── model.timber.json    # Serialized IR
├── audit_report.json    # Compilation audit trail
├── model_info.json      # Registry metadata
├── CMakeLists.txt
└── Makefile
```

## Overwriting Models

Loading with the same name overwrites the existing model:

```bash
timber load model_v2.json --name fraud-detector
# Replaces the previous "fraud-detector"
```

## Custom Store Location

By default, models are stored in `~/.timber/`. Override with:

```bash
export TIMBER_HOME=/opt/timber
timber load model.json --name prod-model
# Cached in /opt/timber/models/prod-model/
```

## Programmatic Loading

```python
from timber.store import ModelStore

store = ModelStore()
info = store.load_model("model.json", name="my-model")
print(f"Loaded: {info['name']}, {info['n_trees']} trees")
```
