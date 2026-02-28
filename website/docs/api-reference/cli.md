---
sidebar_position: 1
title: CLI Reference
---

# CLI Reference

All Timber commands and their options.

## `timber load <model_path>`

Compile and cache a model locally.

```bash
timber load model.json
timber load model.json --name fraud-detector
timber load pipeline.pkl --format sklearn
```

| Option | Default | Description |
|--------|---------|-------------|
| `--name NAME` | Filename stem | Name to register the model as |
| `--format FMT` | Auto-detect | Force input format: `xgboost`, `lightgbm`, `sklearn`, `catboost`, `onnx` |

**What it does:**
1. Auto-detects (or uses forced) format
2. Parses model into Timber IR
3. Runs 6 optimization passes
4. Emits C99 source code
5. Compiles a native shared library (`.so` / `.dylib`)
6. Caches everything in `~/.timber/models/<name>/`

## `timber serve <name>`

Serve a loaded model over HTTP.

```bash
timber serve my-model
timber serve my-model --port 8080
timber serve my-model --host 127.0.0.1 --port 9000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port PORT` | `11434` | Bind port |

The server exposes:
- `POST /api/predict` — run inference
- `POST /api/generate` — alias for predict (Ollama compat)
- `GET /api/models` — list loaded models
- `GET /api/model/:name` — model metadata
- `GET /api/health` — health check

## `timber list`

List all cached models in a table.

```bash
timber list
```

Output:
```
NAME            TREES  FEATURES  FRAMEWORK  SIZE
breast-cancer   50     30        xgboost    58.0 KB
fraud-detector  100    45        lightgbm   112.3 KB
```

## `timber remove <name>`

Remove a cached model and all its artifacts.

```bash
timber remove my-model
```

## `timber compile`

Compile a model to C99 source without caching in the store.

```bash
timber compile --model model.json --out ./dist/
timber compile --model model.json --target targets/x86_64_avx2.toml --out ./dist/
timber compile --model model.json --calibration-data train.csv --out ./dist/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | Required | Model artifact path |
| `--format FMT` | Auto-detect | Input format hint |
| `--target PATH` | `x86_64_generic` | Target spec TOML file |
| `--out DIR` | `./dist` | Output directory |
| `--calibration-data PATH` | None | CSV for branch sorting pass |

## `timber inspect <model_path>`

Print model summary without compiling.

```bash
timber inspect model.json
```

## `timber validate`

Validate compiled output against reference predictions.

```bash
timber validate --artifact ./dist/ --reference model.json --data test.csv
```

| Option | Default | Description |
|--------|---------|-------------|
| `--artifact DIR` | Required | Compiled artifact directory |
| `--reference PATH` | Required | Original model file |
| `--data PATH` | Required | Validation CSV |
| `--tolerance FLOAT` | `1e-5` | Max allowed absolute error |

## `timber bench`

Benchmark inference performance.

```bash
timber bench --artifact ./dist/ --data test.csv --batch-sizes 1,10,100,1000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMBER_HOME` | `~/.timber` | Model store root directory |
| `CC` | `gcc` | C compiler to use for compilation |
