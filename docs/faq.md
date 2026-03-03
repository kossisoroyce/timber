# FAQ

Answers to common questions about Timber.

---

## Installation & Setup

**Q: What do I need installed before using Timber?**

Python 3.10+ and a C compiler (`gcc` or `clang`) that is on your `PATH`.
On Ubuntu/Debian: `sudo apt install gcc`. On macOS: install Xcode Command Line
Tools with `xcode-select --install`. On Windows: use WSL2 or install MinGW.

```bash
gcc --version   # verify gcc is available
pip install timber-compiler
```

**Q: Does Timber work on Windows?**

Natively compiled inference is not yet tested on native Windows. Use WSL2 for
full support. The Python-only parts (parsing, IR, serialization) work anywhere.

**Q: What optional dependencies are there?**

```bash
pip install "timber-compiler[serve]"   # uvicorn (production HTTP)
pip install "timber-compiler[full]"    # all framework deps
```

The core package includes XGBoost, LightGBM, scikit-learn, and ONNX parsers.
CatBoost requires `catboost` installed separately.

---

## Models & Formats

**Q: Which model formats are supported?**

| Framework | Format | Save command |
|-----------|--------|--------------|
| XGBoost | `.json` | `booster.save_model("model.json")` |
| LightGBM | `.txt` | `booster.save_model("model.txt")` |
| scikit-learn | `.pkl` | `pickle.dump(model, f)` |
| CatBoost | `.json` | `model.save_model("m.json", format="json")` |
| ONNX | `.onnx` | `f.write(model.SerializeToString())` |

**Q: My model file is `.json` — how does Timber know if it's XGBoost or CatBoost?**

Timber inspects the file content:
- Contains `"learner"` key → XGBoost
- Contains `"oblivious_trees"` key → CatBoost

Use `--format catboost` or `--format xgboost` to override auto-detection.

**Q: Does Timber support XGBoost binary format (`.ubj`, `.model`)?**

No. Save your model in JSON format: `booster.save_model("model.json")`.

**Q: Does Timber support CatBoost native binary format?**

No. Export to JSON: `model.save_model("model.json", format="json")`.

**Q: Does Timber support regression models?**

Yes. `reg:squarederror`, `reg:logistic`, and LightGBM regression objectives are
all supported. `outputs[0]` is the raw predicted value for regression.

**Q: Does Timber support multiclass classification?**

Yes. `multi:softprob` and `multi:softmax` are fully supported. For K classes,
`outputs` has shape `[n_samples, K]` with per-class probabilities (sum = 1).

**Q: Does Timber support ranking models (LTR)?**

Parsing is supported (`rank:pairwise`, `rank:ndcg`). The output is a raw
ranking score. Softmax/sigmoid are not applied for ranking objectives.

**Q: Can I use a model trained on GPU?**

Yes. Timber reads the saved model file, not the training device. Train on GPU,
save to JSON, load with `timber load`.

**Q: My scikit-learn Pipeline has a `StandardScaler` — will that work?**

Yes. Timber parses the scaler from the pickle and represents it as a
`ScalerStage` in the IR. The optimizer's **pipeline_fusion** pass then folds
the scaler into the tree thresholds, eliminating it from the generated C
entirely. The output is numerically identical to sklearn's prediction.

**Q: What sklearn estimators are supported?**

- `GradientBoostingClassifier` / `GradientBoostingRegressor`
- `RandomForestClassifier` / `RandomForestRegressor`
- `ExtraTreesClassifier` / `ExtraTreesRegressor`
- `DecisionTreeClassifier` / `DecisionTreeRegressor`
- `Pipeline` with `StandardScaler`, `MinMaxScaler`, `SimpleImputer` as steps

Custom or uncommonly-wrapped estimators may require a custom frontend parser.

---

## Inference & Performance

**Q: How fast is Timber vs Python XGBoost?**

On Apple M2 Pro, 50-tree binary classifier, 30 features, single sample:
- Timber native: ~2 µs
- Python XGBoost: ~670 µs
- Speedup: ~336×

This is in-process latency, not HTTP round-trip. Network adds ~50–200 µs.

**Q: What is the output format for binary classification?**

`outputs[0]` ∈ (0, 1) — probability of the positive class. Same as
`predict_proba()[:, 1]` in sklearn or `booster.predict()` in XGBoost.

**Q: How do I get the predicted class (not probability)?**

Threshold at 0.5: `predicted_class = int(outputs[0] > 0.5)`.

**Q: Are predictions numerically identical to the source framework?**

Within 1e-4 by default. Use `timber validate` to check:

```bash
timber validate --artifact ./dist/ --reference model.json --tolerance 1e-5
```

Differences arise from float32 threshold quantization (optimizer pass 3).
Disable with `--no-quantize` if exact float64 matching is required.

**Q: Does the HTTP server support concurrent requests?**

The compiled model context is read-only after initialization, so the C
inference code is thread-safe. The stdlib HTTP server processes requests
serially in the current version. For concurrent workloads, run multiple
server instances behind a load balancer, or use `--serve uvicorn` (roadmap).

**Q: What is the maximum request size?**

64 MB. Requests exceeding this return HTTP 413. For very large batches,
split into multiple requests or use the Python `TimberPredictor` API directly.

---

## Deployment

**Q: Can I deploy the compiled model without Python?**

Yes. After `timber compile --out ./dist/`, the output is pure C99:

```bash
gcc -O2 -shared -fPIC -o libtimber_model.so model.c model_data.c -lm
```

No Python, no Timber, no framework at runtime.

**Q: Can I embed the generated C in a C++ project?**

Yes. `model.h` uses `extern "C"` guards:

```cpp
#include "model.h"
// Use timber_init, timber_infer_single, etc. normally
```

**Q: Can I target ARM or embedded devices?**

The generated C99 is portable and compiles on ARM with `arm-linux-gnueabihf-gcc`
or similar cross-compilers. Cortex-M and RISC-V profiles are on the roadmap.

**Q: Can I deploy to the browser?**

Yes, via the WebAssembly backend:

```python
from timber.codegen.wasm import WasmEmitter
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")
files = WasmEmitter(ir).emit()
# files["model.wat"]        — WebAssembly Text Format
# files["timber_model.js"]  — JavaScript bindings
```

**Q: How do I change the HTTP port?**

```bash
timber serve my-model --port 8080
```

**Q: How do I bind to localhost only (not expose externally)?**

```bash
timber serve my-model --host 127.0.0.1
```

---

## Model Store & Registry

**Q: Where are models stored?**

`~/.timber/models/<name>/` by default. Override with `TIMBER_HOME`:

```bash
export TIMBER_HOME=/opt/timber
timber load model.json --name prod-model
```

**Q: How do I update a model?**

Run `timber load` with the same `--name`. The existing artifact is replaced.

**Q: How do I back up all my models?**

Copy `~/.timber/` or set `TIMBER_HOME` to a versioned directory.

**Q: What is in the model directory?**

```
~/.timber/models/my-model/
├── model.c              # C99 inference logic
├── model.h              # public API header
├── model_data.c         # static tree data arrays
├── model.timber.json    # serialized IR (for inspection)
├── libtimber_model.so   # compiled shared library
└── audit_report.json    # compilation audit trail
```

**Q: Can I share a compiled model directory?**

Yes. Copy the directory to another machine with the same OS and architecture.
The `.so` is not portable across architectures — recompile with
`timber load` on the target machine for cross-architecture deployment.

---

## Errors & Troubleshooting

**Q: `gcc: command not found` error on `timber load`**

Install gcc: `sudo apt install gcc` (Ubuntu) or `xcode-select --install` (macOS).

**Q: `Format not recognized` on `timber load`**

Use `--format` to specify: `timber load model.json --format xgboost`.

**Q: `expected N features, got M` from `/api/predict`**

Your input has the wrong number of features. Check `timber inspect <name>` for
the expected feature count, then ensure your input array shape is correct.

**Q: `Model 'x' not found` from CLI**

Run `timber list` to see registered names. Model names are case-sensitive and
only allow `[a-z0-9_-]` characters.

**Q: Predictions differ slightly from the source framework**

Run `timber validate` to quantify the difference. If the error is above your
tolerance, it may be due to threshold quantization — check the audit report
for which optimizer passes were applied.

**Q: `timber serve` exits immediately**

The model may not exist. Check with `timber list`. If loading from a URL,
check the URL is accessible. Add `--port 8080` to avoid port conflicts.

**Q: Compilation fails with a C compiler warning**

Timber generates `-Wall -Werror`-clean C99 (verified in CI). If you see
compiler errors, it may be a gcc version incompatibility — please open an issue.

---

## Development & Contributing

**Q: How do I run the test suite?**

```bash
git clone https://github.com/kossisoroyce/timber.git
cd timber
pip install -e ".[dev]"
pytest tests/ -v          # 146 tests
ruff check timber/        # lint
```

**Q: How do I add support for a new framework?**

1. Create `timber/frontends/<framework>_parser.py`
2. Implement `parse_<framework>_model(path: str) -> TimberIR`
3. Register in `timber/frontends/auto_detect.py`
4. Add tests in `tests/`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full details.

**Q: What Python versions are tested?**

Python 3.10, 3.11, and 3.12 on Ubuntu (latest) and macOS (latest) via
GitHub Actions CI on every push to `main`.

**Q: Is Timber production-ready?**

Timber is at `0.2.0` (alpha). The C99 backend and core CLI are stable. The
WASM backend, MISRA-C emitter, and remote model registry are in active
development. Use in production with appropriate validation via `timber validate`.
