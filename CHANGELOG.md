# Changelog

All notable changes to Timber are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)  
Versioning: [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

---

## [0.2.0] — 2026-03-03

### Fixed

- **XGBoost 3.1+ multiclass `base_score`** — XGBoost 3.1+ serializes per-class base scores as a comma-separated bracket string (e.g. `'[-3.95E-2,1.97E-1,-1.57E-1]'`). Timber previously discarded all but the first value and initialized all class accumulators to `0.0`, producing systematic probability errors up to 7.5%. Now the full per-class vector is parsed and applied correctly.

### Added

- `per_class_base_scores: list[float]` field on `TreeEnsembleStage` IR with full serialisation/deserialisation support
- `_parse_base_score_list()` helper in `xgboost_parser.py` for robust base_score parsing across XGBoost versions
- `TIMBER_CLASS_BASE_SCORES[]` static array emitted in `model_data.c` for multiclass models
- Comprehensive documentation overhaul — all four docs pages expanded and `llms.txt` added at repo root
- `mkdocs.yml` for docs site structure

### Changed

- Version bumped from `0.1.0` → `0.2.0`
- README completely rewritten: live terminal demo, compiler pipeline diagram, full CLI and API reference tables, runtime comparison feature matrix, roadmap with status indicators

---

## [0.1.0] — 2026-03-03

### Added

- **`timber load <file|url>`** — compile and cache a model from a local file or HTTPS URL with stage-by-stage rich terminal output
- **`timber pull <url>`** — dedicated URL pull command with streaming download progress bar and local cache (`~/.timber/cache/`)
- **`timber serve <name|file|url>`** — pull, compile, and serve in one step; shows serving panel with endpoint, API table, and curl example
- **`timber list`** — rich table output with framework, trees, features, size, compiled status
- **`timber remove <name>`** — styled success/error output
- **`timber compile`** — compile model to C99 artifact with optimizer passes
- **`timber inspect`** — model summary without compiling
- **`timber validate`** — compare compiled artifact against reference model
- **`timber bench`** — latency benchmarks (P50/P95/P99) across batch sizes
- **`timber/ui.py`** — reusable rich terminal UI components (header, section rules, ok/skip/fail lines, serving panel)
- **`timber/downloader.py`** — HTTP(S) streaming download with rich progress bar, atomic temp-file write, URL-keyed local cache, `--force` re-download
- **`LoadCallbacks`** dataclass in `timber/store.py` — per-stage progress hooks wired through `load_model()`
- Optimizer pipeline: 6 passes (dead-leaf elimination, constant-feature detection, threshold quantization, frequency branch sort, pipeline fusion, vectorization analysis)
- Code generation backends: C99 (primary), WebAssembly, MISRA-C
- Supported model formats: XGBoost JSON, LightGBM text, scikit-learn pickle, ONNX TreeEnsemble, CatBoost JSON
- Hardware target specs via TOML (`targets/`)
- Full technical paper (`paper/timber_paper.pdf`)
- CI/CD: GitHub Actions matrix across Python 3.10/3.11/3.12 on Ubuntu and macOS
- PyPI release workflow with OIDC Trusted Publishing

### Security

- Registry writes are now atomic (write to `.json.tmp`, then `rename()`) — prevents corruption on crash
- Model name sanitization uses strict allowlist (`[a-z0-9_-]`) preventing path traversal
- HTTP server enforces 64 MB body size limit, returns HTTP 413 on oversize requests
- `Content-Length` header parse errors return HTTP 400 instead of crashing

### Fixed

- `Content-Length` header `ValueError` in inference server now returns 400 instead of 500
- Banner version hardcoded to `v0.1.0` — now reads `timber.__version__` at runtime
- File handle leak in `on_emit_done` callback (missing `with` block on `open()`)
- Partial downloads no longer cached — atomic `.part` file renamed on success only
- `store.get_model_dir()` returning `None` no longer crashes `serve` with `TypeError`
- Dead conditional branch in `TimberPredictor.predict()` removed
- Temp directory created by `TimberPredictor.from_model()` now cleaned up via `atexit`
- `model.h` missing from artifact directory now raises a clear `FileNotFoundError`
- `bench --warmup-iters` value was silently capped at 100; now uses the full user-supplied value
- `timber list` model names printed before table (cosmetic ordering); names now appear after

[Unreleased]: https://github.com/kossisoroyce/timber/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kossisoroyce/timber/releases/tag/v0.1.0
