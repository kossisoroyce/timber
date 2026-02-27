# Contributing to Timber

Thanks for your interest in contributing to Timber! This guide will help you get set up.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/kossisoroyce/timber.git
cd timber

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Verify everything works
pytest tests/ -v
```

**Requirements:**

- Python 3.10+
- A C compiler (`gcc` or `clang`) for shared library compilation
- `pytest` for testing (installed with `[dev]` extras)

## Running Tests

```bash
# Full suite (144 tests)
pytest tests/ -v

# Specific test file
pytest tests/test_store.py -v

# Specific test
pytest tests/test_store.py::TestModelStore::test_load_model -v

# With coverage
pytest tests/ --cov=timber --cov-report=html
```

## Project Structure

```
timber/
├── ir/                  # Intermediate Representation (data model)
├── frontends/           # Model format parsers
│   ├── xgboost_parser.py
│   ├── lightgbm_parser.py
│   ├── sklearn_parser.py
│   ├── catboost_parser.py
│   ├── onnx_parser.py
│   └── auto_detect.py
├── optimizer/           # IR optimization passes (6 passes)
│   ├── pipeline.py      # Pass orchestration
│   ├── dead_leaf.py
│   ├── constant_feature.py
│   ├── threshold_quant.py
│   ├── branch_sort.py
│   ├── pipeline_fusion.py
│   └── vectorize.py
├── codegen/             # Code generation backends
│   ├── c99.py           # Primary C99 emitter
│   ├── wasm.py          # WebAssembly emitter
│   └── misra_c.py       # MISRA-C compliance emitter
├── runtime/             # Python ctypes predictor
├── audit/               # Audit report generation
├── store.py             # Local model store (~/.timber/models/)
├── serve.py             # HTTP inference server
└── cli.py               # CLI entry point
```

## Making Changes

### Adding a New Front-End Parser

1. Create `timber/frontends/<framework>_parser.py`
2. Implement a `parse_<framework>_model(path: str) -> TimberIR` function
3. Register it in `timber/frontends/auto_detect.py`:
   - Add format detection in `detect_format()`
   - Add dispatch in `parse_model()`
4. Add tests in `tests/test_<framework>_parser.py`

### Adding an Optimizer Pass

1. Create `timber/optimizer/<pass_name>.py`
2. Implement the pass as a function taking `TimberIR` and returning `TimberIR`
3. Register it in `timber/optimizer/pipeline.py` in the `run()` method
4. Add tests

### Adding a Code Generation Backend

1. Create `timber/codegen/<backend>.py`
2. Follow the pattern of `c99.py` — take a `TimberIR`, return a `dict[str, str]` of filename → content
3. Add tests

## Code Style

- Follow existing code patterns and naming conventions
- Type hints on all function signatures
- Docstrings on public classes and functions
- No additional comments unless they clarify non-obvious logic

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Write tests for your changes
3. Ensure all 144+ tests pass: `pytest tests/ -v`
4. Keep commits focused and well-described
5. Open a PR with a clear description of what and why

## Reporting Issues

Use [GitHub Issues](https://github.com/kossisoroyce/timber/issues) with the provided templates:

- **Bug Report**: Include model format, error message, and steps to reproduce
- **Feature Request**: Describe the use case and proposed approach

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
