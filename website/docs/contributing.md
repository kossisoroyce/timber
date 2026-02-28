---
sidebar_position: 12
title: Contributing
---

# Contributing

We welcome contributions to Timber. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/kossisoroyce/timber.git
cd timber
pip install -e ".[dev]"
pytest tests/ -v  # 144 tests should pass
```

**Requirements:** Python 3.10+, a C compiler (`gcc` or `clang`).

## Project Structure

```
timber/
├── ir/                  # Intermediate Representation
├── frontends/           # 5 model format parsers
│   ├── xgboost_parser.py
│   ├── lightgbm_parser.py
│   ├── sklearn_parser.py
│   ├── catboost_parser.py
│   ├── onnx_parser.py
│   └── auto_detect.py
├── optimizer/           # 6 optimization passes
│   ├── pipeline.py
│   ├── dead_leaf.py
│   ├── constant_feature.py
│   ├── threshold_quant.py
│   ├── branch_sort.py
│   ├── pipeline_fusion.py
│   └── vectorize.py
├── codegen/             # 3 code generation backends
│   ├── c99.py
│   ├── wasm.py
│   └── misra_c.py
├── runtime/             # Python ctypes predictor
├── audit/               # Audit report generation
├── store.py             # Local model store
├── serve.py             # HTTP server
└── cli.py               # CLI entry point
```

## Adding a New Front-End

1. Create `timber/frontends/<framework>_parser.py`
2. Implement `parse_<framework>_model(path) → TimberIR`
3. Register in `auto_detect.py` (detection + dispatch)
4. Add tests in `tests/test_<framework>_parser.py`

## Adding an Optimizer Pass

1. Create `timber/optimizer/<pass_name>.py`
2. Implement as a function: `TimberIR → TimberIR`
3. Register in `timber/optimizer/pipeline.py`
4. Add tests

## Adding a Code Backend

1. Create `timber/codegen/<backend>.py`
2. Follow the pattern of `c99.py`: take IR, return `dict[str, str]`
3. Add tests

## Running Tests

```bash
pytest tests/ -v                              # Full suite
pytest tests/test_store.py -v                 # Specific file
pytest tests/ --cov=timber --cov-report=html  # Coverage
```

## Pull Request Process

1. Fork the repo, create a branch from `main`
2. Write tests for new functionality
3. All 144+ tests must pass
4. Keep commits focused
5. Open a PR with a clear description

## Code Style

- Type hints on all function signatures
- Docstrings on public classes/functions
- Follow existing patterns

## License

Contributions are licensed under Apache-2.0.
