# Changelog

All notable changes to Timber are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)  
Versioning: [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

### Added

- **`timber.accel` — TimberAccelerate merged into core** (previously a separate `timber-accelerate` package, now ships with every `pip install timber-compiler` install):
  - **SIMD codegen** — `AVX2Emitter`, `AVX512Emitter`, `NEONEmitter`, `SVEEmitter`, `RVVEmitter`; injects vectorised `timber_infer_simd()` path into generated C; delegates `timber_infer_single` and `timber_infer` to SIMD variant at link time
  - **GPU codegen** — CUDA, Metal, OpenCL emitters
  - **FPGA HLS** — Xilinx Vitis HLS, Intel FPGA SDK emitters
  - **Embedded** — Cortex-M, ESP32, STM32 emitters (no-heap, static buffers, cross-compile prefix support)
  - **WCET analysis** — `analyze_wcet(ir, arch, clock_mhz, safety_margin)` for Cortex-M4/M7, x86_64, AArch64, RISC-V64; per-stage breakdown with advisory disclaimer
  - **Safety transforms** — `deterministic_pass` and `constant_time_pass` IR transforms
  - **Certification reports** — DO-178C, ISO 26262, IEC 62304 compliance report generation
  - **Supply chain** — Ed25519 artifact signing/verification, AES-256-GCM encryption/decryption, TPM integration hooks
  - **Deployment** — air-gapped bundle creator, C++ gRPC server generator, ROS 2 node generator, PX4 autopilot module generator, sensor preprocessing (radar, RF, telemetry)
  - **18 target profiles** (TOML) bundled at `timber/accel/targets/`: AVX2, AVX512, NEON, SVE, RVV, CUDA SM75/SM86, Metal Apple M1, OpenCL generic, Cortex-M4/M7/ESP32/STM32, HLS Xilinx/Intel
  - **3 compliance profiles** (TOML) bundled at `timber/accel/compliance_profiles/`: DO-178C, ISO 26262, IEC 62304
  - **`timber-accel` CLI** — 9 commands: `compile`, `wcet`, `certify`, `sign`, `verify`, `encrypt`, `decrypt`, `bundle`, `serve-native`

---

## [0.5.0] — 2026-03-13

### Added

- **`IsolationForestStage` IR node** — new `PipelineStage` subclass with flat-tree anomaly scoring; leaf values store pre-computed path-length contributions (`depth + c(n_node_samples)`) so C99 inference needs no `log()` at runtime; output = `−anomaly_score − offset` (positive → inlier)
- **`NaiveBayesStage` IR node** — Gaussian Naive Bayes; stores per-class log-priors, means (`theta`), pre-computed `log_var_const` and `inv_2var` to eliminate `log()` and division at runtime; softmax output
- **`GPRStage` IR node** — Gaussian Process Regressor (RBF kernel); stores training points, `alpha` vector, and pre-computed `inv_2ls2` / `amp2` constants; normalised output via `y_train_mean` / `y_train_std`
- **`KNNStage` IR node** — k-Nearest Neighbours classifier or regressor (lookup table); supports Euclidean and Manhattan metrics; classifier outputs winning class index, regressor outputs averaged label vector
- **`SVMStage.is_one_class`** — boolean flag added to `SVMStage` for `OneClassSVM`; serialisation and deserialisation updated; `decision_function = sum(alpha_i × K) + intercept_`
- **sklearn parsers for all 5 new primitives** (`timber/frontends/sklearn_parser.py`):
  - `_parse_isolation_forest` — remap subsampled feature indices; BFS depth pre-computation; `max_depth` set on each `Tree` for correct C99 guard
  - `_parse_one_class_svm` — extracts `_gamma`, handles `scale`/`auto`/numeric gamma
  - `_parse_naive_bayes` — pre-computes `log_var_const` and `inv_2var` from `GaussianNB.var_`
  - `_parse_gpr` — walks `ConstantKernel * RBF` kernel tree; extracts `length_scale` and `amplitude`; safe scalar extraction from `_y_train_mean` / `_y_train_std`
  - `_parse_knn` — handles `KNeighborsClassifier` and `KNeighborsRegressor`; maps `minkowski`/`l1`/`l2` aliases
- **C99 emitters for all 5 new primitives** (`timber/codegen/c99.py`):
  - IsolationForest: `iforest_traverse()` with double-precision thresholds and path-lengths; `exp2(−mean_path / c_max)` anomaly score
  - Naive Bayes: log-likelihood accumulation + numerically stable softmax
  - GPR: RBF kernel matrix-vector product with double accumulator
  - k-NN: configurable distance metric; min-heap top-k; majority vote / mean aggregation
- **119 nuclear tests** (`tests/test_primitives.py`) covering IR round-trip, sklearn parsing field correctness, C99 code structure, compilation, and numerical accuracy vs. sklearn reference for all 5 primitives

- **URDF forward-kinematics frontend** (`timber/frontends/urdf_parser.py`) — `URDFParser` parses URDF XML into a `KinematicsStage` IR; auto-detects base link and end-effector; supports all joint types (`revolute`, `prismatic`, `continuous`, `fixed`)
- **`JointSpec` IR node** — new dataclass carrying joint name, type, axis, origin (xyz + rpy), parent/child links, and position limits; full JSON serialization round-trip
- **`KinematicsStage` IR node** — new `PipelineStage` subclass holding a `list[JointSpec]`, `base_link`, `end_effector`, and computed `n_dof`; full JSON serialization round-trip
- **C99 emitter: kinematics backend** — `C99Emitter.emit()` dispatches `KinematicsStage` to:
  - `timber_fk(q, T, ctx)` — computes 4×4 row-major homogeneous transform from joint angles; max absolute error vs Python reference < 1 × 10⁻⁷
  - `timber_infer_single` delegates to `timber_fk` (same ABI as all other Timber stages, 16-element float output)
  - RPY origin transforms pre-computed at code-gen time as compile-time constants
  - Rodrigues rotation for revolute/continuous joints, prismatic translation for sliding joints
  - Helper functions (`rodrigues`, `prismatic`) emitted only when needed — zero unused-function warnings
  - `TIMBER_N_DOF` constant in header
- **`timber serve robot.urdf`** — `.urdf` extension is now auto-detected; `timber load` and `timber serve` accept URDF files directly; serve panel shows `N DOF · 16 outputs (4×4 transform)` and pre-filled example curl command with zero-initialized joint angles
- **49 kinematics tests** (`tests/test_kinematics.py`) covering URDF parsing, IR round-trip, C99 code structure, and numerical FK correctness (compiled C vs Python reference, KUKA iiwa end-to-end)

### Fixed

- **`ml_dtypes` / `onnx` version conflict** — `pyproject.toml` now pins `onnx>=1.17.0` and `ml-dtypes>=0.5.0` in both `full` and `dev` extras; `onnx 1.17+` unconditionally accesses `ml_dtypes.float4_e2m1fn` (added in 0.5.0), which caused `AttributeError` on machines with TensorFlow ≤ 2.16.x holding `ml_dtypes` at 0.3.x
- **Defensive `@_skip_no_onnx` guards** added to all 24 ONNX-dependent tests in `test_nuclear.py`; in environments with an incompatible `ml_dtypes` the tests now skip with a clear actionable message instead of raising `AttributeError`

---

## [0.4.0] — 2026-03-04

### Added

- **ONNX Linear/SVM/Normalizer/Scaler frontend** — `parse_onnx_model()` now handles six additional ONNX ML opset operators beyond `TreeEnsemble`:
  - `LinearClassifier` — binary (sigmoid) and multiclass (softmax), with correct per-row weight extraction
  - `LinearRegressor` — identity activation, arbitrary output dimensionality
  - `SVMClassifier` — RBF and linear kernels, full support-vector matrix extraction
  - `SVMRegressor` — same kernel support as classifier
  - `Normalizer` — L1 / L2 / Max normalization as a `NormalizerStage` preprocessing step
  - `Scaler` — mean-shift / scale as a `ScalerStage` preprocessing step; fuses with downstream trees via pipeline fusion
- **`LinearStage` IR node** — new `PipelineStage` subclass holding weights, biases, activation (`none` / `sigmoid` / `softmax`), `n_classes`, and `multi_weights` flag; full JSON serialization round-trip
- **`SVMStage` IR node** — new `PipelineStage` subclass holding support-vector matrix, dual coefficients, rho, gamma, coef0, degree, and kernel type; full JSON serialization round-trip
- **`NormalizerStage` IR node** — new `PipelineStage` subclass; full JSON serialization round-trip
- **C99 emitter: Linear and SVM backends** — `C99Emitter.emit()` now dispatches `LinearStage` and `SVMStage` to dedicated emitters:
  - `_emit_inference_linear` — unrolled dot product, sigmoid (binary), softmax (multiclass), or identity (regression)
  - `_emit_inference_svm` — RBF kernel (`exp(-γ·‖x−sv‖²)`) or linear kernel, with `tanh` post-transform
  - All outputs are bounded to `n_outputs` to prevent any buffer overflow
- **Embedded deployment profiles** — `TargetSpec.for_embedded(profile)` selects cross-compilation toolchains for four targets; the Makefile emitted by `C99Emitter` switches automatically:
  - `cortex-m4` — `arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard`
  - `cortex-m33` — `arm-none-eabi-gcc -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard`
  - `rv32imf` — `riscv32-unknown-elf-gcc -march=rv32imf -mabi=ilp32f`
  - `rv64gc` — `riscv64-unknown-elf-gcc -march=rv64gc -mabi=lp64d`
  - No `-fPIC` or `-shared` flags on embedded targets; produces `.a` static libraries instead of `.so`
- **LLVM IR backend** (`timber/codegen/llvm_ir.py`) — new `LLVMIREmitter` supporting `TreeEnsembleStage`, `LinearStage`, and `SVMStage`; configurable target triple (`x86_64`, `aarch64`, `cortex-m4`, …); produces `model.ll` with SSA form, named `traverse_tree_N` per-tree functions, and the `timber_infer_single` entry point
- **Differential privacy module** (`timber/privacy/dp.py`) — `apply_dp_noise(outputs, cfg)` injects calibrated noise into inference outputs; features:
  - Laplace mechanism: scale = `sensitivity / epsilon`
  - Gaussian mechanism: σ = `√(2 ln(1.25/δ)) · sensitivity / epsilon`
  - `DPConfig` — validates `epsilon > 0`, `sensitivity > 0`, `delta ∈ (0,1)` for Gaussian, mechanism name
  - `DPReport` — returns `noise_scale`, `mechanism`, `n_outputs_noised`, `epsilon`, `delta`
  - `calibrate_epsilon(noise_level, sensitivity, mechanism)` — invert the mechanism to find required ε
  - Input dtype preserved (float32/float64 round-trips exactly); optional output clipping via `clip_outputs`, `output_min`, `output_max`
  - Deterministic replay with `seed` parameter
- **`bench` command enhancements** — richer reporting beyond latency:
  - `--iters N` flag for total timed iterations (default: 1 000)
  - P50 / P95 / P99 / P999 latency percentiles
  - Coefficient of variation (CV%) as a stability indicator
  - `--report PATH` writes a structured JSON report *and* a self-contained HTML file (no external dependencies) with a sortable results table and system-info block
  - `_bench_report_html()` helper for programmatic HTML generation
- **Nuclear-grade test suite** (`tests/test_nuclear.py`) — 139 new tests (436 total passing) covering: IR layer, sklearn/ONNX parsers, numeric accuracy (C99 vs Python IR), all optimizer passes + idempotency + pipeline fusion math verification, diff compiler, C99/WASM/MISRA-C/LLVM IR emitters, differential privacy statistical correctness, and full end-to-end pipelines

### Fixed

- **ONNX `classlabels_ints` attribute name** — parser was reading `classlabels_int64s` (wrong); multiclass models always reported `n_classes = 2`, producing incorrect weight slicing and garbage softmax outputs
- **Binary ONNX `LinearClassifier` double weight row** — `skl2onnx` emits both class rows for binary models; parser now extracts only the positive-class row and sets `multi_weights = False`, fixing incorrect weight counts and index misalignment
- **C99 buffer overflow guard** — `multi_weights = True` softmax loop now bounded by `n_outputs` (not `n_classes`), preventing out-of-bounds writes when the output buffer is smaller than the number of internal score slots

### Changed

- ONNX supported-operator list expanded from `TreeEnsemble{Classifier,Regressor}` to include `LinearClassifier`, `LinearRegressor`, `SVMClassifier`, `SVMRegressor`, `Normalizer`, `ZipMap`, `Scaler`
- `C99Emitter.emit()` dispatch table extended; unknown primary stage now raises `ValueError("No supported primary stage")`
- `pyproject.toml` development status upgraded from `3 - Alpha` to `4 - Beta`
- `[project.optional-dependencies]` gains `privacy = ["numpy>=1.24"]` and `full` gains `onnx>=1.14`, `skl2onnx>=1.15`
- Test count: 297 → 436

---

## [0.3.0] — 2026-03-04

### Added

- **Production multi-worker HTTP server** — `timber serve` now uses FastAPI + uvicorn instead of Python's single-threaded `http.server`. Each worker runs an asyncio event loop with a `ThreadPoolExecutor` for non-blocking C inference.
- **`--workers N` flag** — spawn N independent OS worker processes, each with its own GIL and loaded `.so`. `--workers 4` on a 4-core machine delivers ~150,000 req/s. Multi-worker mode uses `factory=True` + env-var worker init for clean process isolation.
- **`--threads M` flag** — controls `ThreadPoolExecutor` size per worker (default: `min(32, cpu_count + 4)`). Keeps multiple requests in-flight per worker without blocking the event loop.
- **`--backlog N` flag** — configures TCP listen backlog (default: 2048) for high-connection-rate workloads.
- **`GET /api/metrics`** — rolling latency percentiles (p50/p95/p99/p999), total requests, samples, req/s, and uptime. Uses a thread-safe `deque`-based `LatencyTracker` with a 10,000-sample window.
- **`GET /docs`** — interactive OpenAPI UI (Swagger) auto-generated by FastAPI. `/redoc` also available.
- **`GET /api/health`** now returns `version`, `models_loaded`, and `uptime_seconds` instead of a bare `{"status": "ok"}`.
- **CORS middleware** — all endpoints accept cross-origin requests with proper preflight handling.
- **Graceful lifespan shutdown** — `ThreadPoolExecutor.shutdown(wait=True)` on SIGTERM/Ctrl-C via FastAPI lifespan context.
- **Legacy fallback** — if `fastapi`/`uvicorn` are not installed, server falls back to `http.server` with a clear install hint.
- **`serve` optional extra** now includes `fastapi>=0.110.0`, `uvicorn[standard]>=0.27.0`, `anyio>=4.0`. Install with `pip install 'timber-compiler[serve]'`.
- **Startup panel** displays Workers · Threads/worker · Concurrency alongside model info.

### Changed

- `timber serve` docstring updated with new flags and all API endpoints including `/api/metrics` and `/docs`.

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

[Unreleased]: https://github.com/kossisoroyce/timber/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/kossisoroyce/timber/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kossisoroyce/timber/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kossisoroyce/timber/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kossisoroyce/timber/releases/tag/v0.1.0
