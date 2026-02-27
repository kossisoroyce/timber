# Timber

**Ollama for classical ML models.**

Timber compiles trained tree-based models (XGBoost, LightGBM, scikit-learn, CatBoost, ONNX) into optimized native code and serves them over a local HTTP API — just like Ollama does for LLMs, but for small models.

No Python runtime at inference time. Sub-microsecond latency. One command to load, one command to serve.

## Quick Start

```bash
pip install timber
```

### Load a model

```bash
# Load any supported model — Timber auto-detects the format
timber load model.json
timber load model.json --name fraud-detector
timber load model.pkl --format sklearn
```

### Serve it

```bash
timber serve fraud-detector
```

```
  _____ _           _
 |_   _(_)_ __ ___ | |__   ___ _ __
   | | | | '_ ` _ \| '_ \ / _ \ '__|
   | | | | | | | | | |_) |  __/ |
   |_| |_|_| |_| |_|_.__/ \___|_|

  Classical ML Inference Server v0.1.0

  Listening on http://0.0.0.0:11434
  Model:    fraud-detector
  Trees:    100
  Features: 30
```

### Run inference

```bash
curl http://localhost:11434/api/predict \
  -d '{"model": "fraud-detector", "inputs": [[1.0, 2.0, 3.0, ...]]}'
```

```json
{
  "model": "fraud-detector",
  "outputs": [0.97],
  "n_samples": 1,
  "latency_us": 0.8,
  "done": true
}
```

### Manage models

```bash
timber list                    # list all loaded models
timber remove fraud-detector   # remove a model
```

```
NAME                      FORMAT        TREES  FEATURES       SIZE  COMPILED
---------------------------------------------------------------------------
fraud-detector            xgboost         100        30    42.1 KB       yes
churn-model               lightgbm         50        18    28.3 KB       yes
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Run inference — `{"model": "name", "inputs": [[...]]}` |
| `/api/generate` | POST | Alias for `/api/predict` (Ollama compat) |
| `/api/models` | GET | List loaded models |
| `/api/model/:name` | GET | Get model info |
| `/api/health` | GET | Health check |

## Supported Formats

| Format | Framework | File Types |
|--------|-----------|------------|
| XGBoost JSON | XGBoost | `.json` |
| LightGBM text | LightGBM | `.txt`, `.model`, `.lgb` |
| scikit-learn pickle | scikit-learn | `.pkl`, `.pickle` |
| ONNX ML opset | ONNX | `.onnx` |
| CatBoost JSON | CatBoost | `.json` |

All formats are auto-detected. Use `--format` to override.

## Advanced: Direct Compilation

For embedding in C/C++ projects without the server:

```bash
# Compile to C99 source
timber compile --model model.json --out ./dist/

# Inspect a model
timber inspect model.json

# Validate compiled output
timber validate --artifact ./dist/ --reference model.json --data test.csv

# Benchmark
timber bench --artifact ./dist/ --data bench.csv
```

### C API

```c
#include "model.h"

TimberCtx* ctx;
timber_init(&ctx);

float inputs[TIMBER_N_FEATURES] = { /* ... */ };
float outputs[TIMBER_N_OUTPUTS];

timber_infer_single(inputs, outputs, ctx);
timber_free(ctx);
```

### Logging Callback

```c
void my_logger(int level, const char* msg) {
    printf("[timber] %s\n", msg);
}

timber_set_log_callback(my_logger);
```

## Compiler Pipeline

```
Model artifact → Front-end parser → Timber IR → Optimizer → Code generator → Native code
```

### Optimizer Passes

1. **Dead Leaf Elimination** — Prune negligible leaves
2. **Constant Feature Detection** — Fold trivial splits
3. **Threshold Quantization** — Classify thresholds for optimal storage
4. **Frequency-Ordered Branch Sorting** — Reorder for branch prediction (with calibration data)
5. **Pipeline Fusion** — Absorb scalers into tree thresholds
6. **Vectorization Analysis** — Identify SIMD batching opportunities

## Architecture

```
timber/
├── ir/                  # Intermediate Representation
├── frontends/           # Model format parsers (xgboost, lightgbm, sklearn, onnx, catboost)
├── optimizer/           # IR optimization passes (6 passes)
├── codegen/             # Code generation (C99, WebAssembly, MISRA-C)
├── runtime/             # Python ctypes predictor
├── store.py             # Local model registry (~/.timber/models/)
├── serve.py             # HTTP inference server
└── cli.py               # CLI (load, serve, list, remove, compile, inspect, ...)
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v          # 144 tests
```

## License

Apache-2.0
