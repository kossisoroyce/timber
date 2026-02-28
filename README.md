# Timber

[![PyPI version](https://img.shields.io/pypi/v/timber-compiler.svg)](https://pypi.org/project/timber-compiler/)
[![Python versions](https://img.shields.io/pypi/pyversions/timber-compiler.svg)](https://pypi.org/project/timber-compiler/)
[![CI](https://github.com/kossisoroyce/timber/actions/workflows/ci.yml/badge.svg)](https://github.com/kossisoroyce/timber/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Ollama for classical ML models.**

Timber compiles trained tree-based models (XGBoost, LightGBM, scikit-learn, CatBoost, ONNX) into optimized native C and serves them over a local HTTP API.

- No Python runtime in the inference hot path
- Native latency (microseconds)
- One command to load, one command to serve

📚 Docs: https://kossisoroyce.github.io/timber/

## Who is this for?

Timber is built for teams that need **fast, predictable, portable inference**:

- **Fraud/risk teams** running classical models in low-latency transaction paths
- **Edge/IoT teams** deploying models to gateways and embedded devices
- **Regulated industries** (finance, healthcare, automotive) needing deterministic artifacts and audit trails
- **Platform/infra teams** replacing Python model-serving overhead with native binaries

## Quick Start

```bash
pip install timber-compiler
```

```bash
# Load any supported model (auto-detected)
timber load model.json --name fraud-detector

# Serve it (Ollama-style workflow)
timber serve fraud-detector
```

```bash
curl http://localhost:11434/api/predict \
  -d '{"model": "fraud-detector", "inputs": [[1.0, 2.0, 3.0, ...]]}'
```

## Supported Formats

| Format | Framework | File Types |
| --- | --- | --- |
| XGBoost JSON | XGBoost | `.json` |
| LightGBM text | LightGBM | `.txt`, `.model`, `.lgb` |
| scikit-learn pickle | scikit-learn | `.pkl`, `.pickle` |
| ONNX ML opset (TreeEnsemble) | ONNX | `.onnx` |
| CatBoost JSON | CatBoost | `.json` |

## Benchmarks (Methodology + Reproducibility)

The 336× claim is measured against Python XGBoost single-sample inference.

### Methodology

- **Hardware:** Apple M2 Pro, 16 GB RAM, macOS (recorded by script)
- **Model:** XGBoost binary classifier, 50 trees, max depth 4, 30 features
- **Dataset:** breast_cancer (sklearn)
- **Warmup:** 1,000 iterations
- **Timed:** 10,000 single-sample predictions
- **Metric:** in-process latency (not HTTP/network round-trip)
- **Baseline:** Python XGBoost (`booster.predict`)

### Reproducible scripts

See [`benchmarks/`](benchmarks/) for:

- `run_benchmarks.py` (Timber vs Python XGBoost + optional ONNX Runtime/Treelite/lleaves)
- `system_info.py` (hardware/software metadata)
- `render_table.py` (markdown table output)

Run:

```bash
python benchmarks/run_benchmarks.py --output benchmarks/results.json
python benchmarks/render_table.py --input benchmarks/results.json
```

## Comparisons

| Runtime | Runtime deps | Typical artifact size | Latency profile | Notes |
| --- | --- | --- | --- | --- |
| Timber | None (generated C99) | ~48 KB (example model) | **~2 µs native call** | Strong fit for edge/embedded and deterministic deployments |
| Python (xgboost/sklearn serving) | Python + framework stack | 50–200+ MB process footprint | 100s of µs to ms | Easy dev loop, high runtime overhead |
| ONNX Runtime | ONNX Runtime libs | MBs to 10s of MBs | usually low 100s of µs | Broad model ecosystem, larger runtime |
| Treelite Runtime | Treelite runtime + compiled artifact | MB-scale runtime + model lib | low-latency when compiled | Great for GBDTs; separate compile/runtime flow |
| lleaves | Python package + LightGBM text model | Python runtime + compiled code | lower than pure Python | LightGBM-focused |

## Limitations / Known Issues

- ONNX support is currently focused on **TreeEnsembleClassifier/Regressor** operators.
- CatBoost support expects **JSON exports** (not native binary formats).
- scikit-learn parser supports major tree estimators and pipelines; uncommon/custom estimator wrappers may fail.
- Pickle parsing follows Python pickle semantics — only load trusted artifacts.
- XGBoost support is JSON-model based. Binary booster formats are not the primary input path.
- Optional benchmark backends (ONNX Runtime, Treelite, lleaves) are skipped unless installed/configured.

## API Endpoints (serve mode)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/predict` | POST | Run inference |
| `/api/generate` | POST | Alias for `/api/predict` (Ollama compat) |
| `/api/models` | GET | List loaded models |
| `/api/model/:name` | GET | Get model metadata |
| `/api/health` | GET | Health check |

## Roadmap

- Improve framework/version compatibility coverage (including more edge-case model exports)
- Broaden ONNX operator support beyond tree ensembles
- Strengthen embedded deployment profiles (ARM Cortex-M / RISC-V presets)
- Add richer benchmark matrices and public reproducibility reports
- Expand safety/regulatory tooling around audit + MISRA-C workflows

## Examples

End-to-end runnable examples live in [`examples/`](examples/):

- `quickstart_xgboost.py`
- `quickstart_lightgbm.py`
- `quickstart_sklearn.py`

They generate model files you can load immediately with `timber load`.

## Paper

Timber includes a full technical paper: [`paper/timber_paper.pdf`](paper/timber_paper.pdf)

### Citation (BibTeX)

```bibtex
@misc{royce2026timber,
  title        = {Timber: Compiling Classical Machine Learning Models to Native Inference Binaries},
  author       = {Kossiso Royce},
  year         = {2026},
  howpublished = {GitHub repository and technical paper},
  institution  = {Electricsheep Africa},
  url          = {https://github.com/kossisoroyce/timber}
}
```

## Community & Governance

- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- Security policy: [`SECURITY.md`](SECURITY.md)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache-2.0
