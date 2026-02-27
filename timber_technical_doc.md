**TIMBER**

Classical ML Inference Compiler

Technical Product Documentation  |  v0.1

*Timber is a compiler that takes trained classical ML model artifacts — XGBoost, LightGBM, scikit-learn, and others — and emits optimized, self-contained inference binaries targeting specific hardware. No Python runtime at inference time. No generic runtime overhead. Just fast, auditable, portable code.*

This document defines the end-to-end technical architecture, compiler pipeline, supported formats, optimization strategy, output targets, and integration surface of Timber.

*Version: 0.1 — Initial Architecture Draft*

*Status: Pre-implementation design specification*

# **1\. Problem Statement**

The modern MLOps ecosystem was built almost entirely around deep learning. Frameworks like TensorFlow Serving, Triton Inference Server, and TorchServe were designed to serve neural networks — and they show it. Classical ML models, which still represent the majority of production workloads in finance, insurance, healthcare, ad tech, and industrial applications, are treated as second-class citizens.

The typical production path for an XGBoost or scikit-learn model today looks like this:

* Train model in Python, serialize to .pkl or framework-native format

* Wrap in a Flask or FastAPI endpoint

* Deploy behind a load balancer with the full Python runtime

* Pay the overhead of CPython, scikit-learn, and pandas on every inference call

This approach carries significant costs:

| Latency | Python function call overhead, interpreter dispatch, and memory allocation add microseconds to milliseconds on every request — unacceptable at high throughput |
| :---- | :---- |
| **Memory** | A Python process serving a 2MB XGBoost model may consume 200MB+ of RAM due to runtime dependencies |
| **Portability** | Deployment requires matching Python versions, dependency trees, and platform-specific wheels — brittle and painful |
| **Auditability** | A .pkl file is not human-readable, not certifiable, and not inspectable by safety reviewers |
| **Edge / Embedded** | Shipping a Python runtime to a microcontroller or ECU is simply not possible |

Timber solves this by treating classical ML inference as a compilation problem, not a runtime problem.

# **2\. Product Overview**

## **2.1 What Timber Is**

Timber is an ahead-of-time (AOT) compiler for classical machine learning models. It ingests a trained model artifact and a target hardware specification, and emits an optimized inference artifact — a shared library, static library, WebAssembly module, or standalone executable — that performs inference with zero runtime dependencies beyond the C standard library.

Timber is not a serving framework, a training library, or an MLOps platform. It occupies the layer between model training and model deployment, analogous to what LLVM does for programming languages — it is the backend that turns high-level model representations into efficient machine code.

## **2.2 Core Design Principles**

* Ahead-of-time over just-in-time. Models are compiled fully before deployment. There is no JIT warmup, no interpreter, no runtime graph rewriting.

* Static shape exploitation. Classical ML models have fixed input dimensionality. Timber exploits this fully — loops are unrolled, strides are hardcoded, bounds checks are eliminated.

* Target-aware code generation. The compiler knows the deployment target at compile time and emits code specifically tuned for that ISA, cache topology, and SIMD width.

* Whole-pipeline compilation. Preprocessing stages — scaling, encoding, imputation — are compiled together with the model into a single fused artifact. No boundary overhead.

* Auditability by design. Every compiled artifact is accompanied by a deterministic compilation report documenting every optimization applied, every tree pruned, and every precision decision made.

## **2.3 Name**

Timber. Decision trees are made of branches and leaves. The compiler turns forests into lumber — raw, optimized material ready to be built with.

# **3\. Supported Input Formats**

Timber defines a canonical internal representation (the Timber IR) and provides front-end parsers for each supported framework format. Adding a new framework means writing a new parser that emits Timber IR — the optimizer and backends are framework-agnostic.

## **3.1 Phase 1 — Initial Target Formats**

| Format | Framework | Model Types |
| :---- | :---- | :---- |
| XGBoost JSON / binary | XGBoost | GBT Classifier, GBT Regressor, GBT Ranker |
| LightGBM model.txt | LightGBM | GBT Classifier, GBT Regressor |
| scikit-learn Pipeline (.pkl) | scikit-learn | RandomForest, GradientBoosting, LogisticRegression, LinearSVC, plus preprocessing stages |
| ONNX (.onnx) | ONNX ML opset | TreeEnsemble operators, Linear operators, preprocessing ops |

## **3.2 Phase 2 — Planned**

* CatBoost model artifacts (.cbm)

* Spark MLlib exported models (via MLeap intermediate format)

* H2O MOJO format

* Custom rule-based models via Timber Rule DSL

# **4\. Timber Intermediate Representation (IR)**

All front-end parsers lower their input to the Timber IR before optimization. The IR is a typed, hierarchical representation of a full inference pipeline. It is serializable, diffable, and inspectable independently of any framework.

## **4.1 IR Structure**

A Timber IR document consists of three top-level sections:

* Pipeline — an ordered sequence of stages. Each stage is either a preprocessing transform or a model.

* Schema — the input and output schema of the full pipeline: field names, types, shapes, and value constraints.

* Metadata — provenance, training framework version, feature importance, and compilation hints.

## **4.2 Stage Types**

| Stage Type | Description | Example |
| :---- | :---- | :---- |
| Scaler | Elementwise affine transform per feature | StandardScaler, MinMaxScaler |
| Encoder | Categorical to numeric transform | OneHotEncoder, OrdinalEncoder, TargetEncoder |
| Imputer | Missing value fill | SimpleImputer (mean/median/mode/constant) |
| TreeEnsemble | Gradient boosted or bagged tree ensemble | XGBoost, LightGBM, RandomForest |
| Linear | Dot product \+ bias with optional nonlinearity | LogisticRegression, LinearSVC |
| Aggregator | Combines multiple stage outputs | Voting ensemble, Stacking |

## **4.3 Tree Representation**

Tree ensembles are represented in the IR as a flat array of nodes. Each node carries:

* A feature index (into the input vector)

* A split threshold (float32 or float16 depending on precision mode)

* Left and right child offsets (relative, enabling SIMD-friendly vectorization)

* A leaf value or leaf distribution (for multi-class classification)

* A depth tag (used for cache-aware traversal scheduling)

This flat layout is different from the pointer-linked tree structures used by XGBoost and LightGBM internally. The conversion is done at parse time and enables the vectorized traversal strategies described in Section 6\.

# **5\. Compiler Pipeline**

*Input model artifact → Front-end parser → Timber IR → Optimizer passes → Code generator → Target artifact*

## **5.1 Front-End Parsing**

Each supported framework has a dedicated parser module. The parser is responsible for:

* Deserializing the model artifact in a format-safe way (handling version differences, missing fields, non-standard encodings)

* Validating model integrity and flagging unsupported constructs

* Lowering the framework-native structure to Timber IR

* Attaching metadata: original framework version, objective function, feature names if available

## **5.2 Optimizer Passes**

The optimizer operates on the Timber IR and applies a sequence of passes. Passes are ordered and may be repeated until a fixpoint is reached. Each pass is independently testable and produces a diff against the previous IR state for auditability.

### **Pass 1: Dead Leaf Elimination**

Prune leaves whose contribution to the final prediction falls below a configurable threshold relative to the largest leaf value in the ensemble. This reduces tree size without meaningful accuracy loss. The threshold is configurable (default: 0.001 relative contribution).

### **Pass 2: Constant Feature Detection**

Features with zero variance in the training data (as recorded in model metadata) are identified. Split nodes gating exclusively on constant features are folded to their dominant branch. The corresponding input column is flagged as ignorable — the code generator will not read it.

### **Pass 3: Threshold Quantization**

Split thresholds are analyzed for precision requirements. Where float32 precision is unnecessary (e.g., features representing integer counts, binary flags, bounded categorical ordinals), thresholds are quantized to float16 or int8. This halves or quarters the size of the threshold array, improving cache utilization.

### **Pass 4: Frequency-Ordered Branch Sorting**

This pass requires a calibration dataset — a representative sample of real inference inputs (minimum 1,000 rows, recommended 10,000+). The pass profiles split node outcomes across the calibration set and reorders each node's children so the more frequently taken branch is evaluated first. On modern CPUs with static branch prediction defaulting to fall-through, this meaningfully reduces mispredictions.

### **Pass 5: Pipeline Fusion**

Preprocessing stages (Scalers, Encoders, Imputers) are analyzed for opportunities to fuse with downstream stages. Where a scaler is followed directly by a tree ensemble, the scaler's per-feature affine transforms are absorbed into the split thresholds — the input is compared to a pre-adjusted threshold, eliminating the scaling step at runtime entirely.

### **Pass 6: Vectorization Layout**

The flat node array is rearranged into a Structure-of-Arrays layout optimized for the target SIMD width. For AVX-512 targets, 16 tree traversals are interleaved — 16 data points traverse all trees in lockstep, with comparisons and conditional moves executed as vector operations. For ARM NEON (128-bit), the interleave width is 4\.

## **5.3 Code Generation**

The code generator lowers optimized Timber IR to target-specific code. Timber supports multiple code generation backends:

| Backend | Output | Use Case |
| :---- | :---- | :---- |
| C99 emitter | Portable C source (.c \+ .h) | Embedded targets, safety-certified environments, human readability |
| LLVM IR emitter | LLVM bitcode (.bc) | Maximum optimization via LLVM backend, x86/ARM/RISC-V targets |
| WASM emitter | WebAssembly module (.wasm) | Browser-side inference, edge workers, serverless |
| Assembly emitter | Hand-tuned x86 ASM (.asm) | Ultra-low-latency paths requiring direct register control |

The LLVM IR emitter is the primary backend for cloud and server deployments. It emits LLVM IR with target feature annotations (e.g., \+avx512f, \+avx512bw) and delegates to LLVM for register allocation, instruction scheduling, and final machine code emission.

The C99 emitter is the primary backend for embedded and safety-critical deployments. It emits strict C99 with no dynamic allocation, no recursion, and no floating-point unless the target explicitly supports it. The output is readable, auditable, and compilable with any C99-compliant toolchain.

# **6\. Inference Execution Model**

## **6.1 Single-Sample Inference**

For latency-optimized single-sample inference (batch size 1), Timber compiles the entire forward pass as a straight-line sequence of operations — no loops over trees, no dynamic dispatch. Each tree is unrolled into a nested conditional structure. The compiler statically determines the maximum depth across all trees and pads shallower trees to match, enabling the code generator to emit uniform basic blocks with no variable-length control flow.

The compiled artifact for a 500-tree XGBoost model with maximum depth 6 executing a single inference consists of approximately 500 × 64 \= 32,000 comparisons and conditional moves. At modern CPU clock speeds this executes in under 10 microseconds on x86.

## **6.2 Batched Inference**

For throughput-optimized batched inference, Timber uses the vectorized traversal strategy developed in Pass 6 of the optimizer. Rather than executing trees one sample at a time, multiple samples traverse each tree in parallel using SIMD comparison instructions.

On AVX-512 hardware, 16 float32 comparisons are executed per instruction. A batch of 16 samples traverses a depth-6 tree in 6 vector comparison steps plus 6 conditional selection steps — 12 total instructions instead of 16 × 6 \= 96 scalar instructions. This is roughly an 8x throughput improvement before accounting for cache effects.

## **6.3 Memory Access Pattern**

The flat node array layout (Section 4.3) is designed for sequential memory access. During batched traversal, the 16 samples simultaneously access the same node — their divergence only appears in which child to follow. The node array is sized to fit in L1 or L2 cache for models of typical depth. For large ensembles, the code generator emits prefetch instructions for the probable next node based on the frequency data from Pass 4\.

## **6.4 Precision Modes**

Timber supports three precision modes, selectable at compile time:

* float32 (default) — Full single-precision throughout. Bit-identical to framework reference implementation.

* float16 — Half-precision thresholds and leaf values. Acceptable accuracy for most tabular tasks. 2x reduction in data working set, improving cache hit rates.

* mixed — float32 arithmetic, float16 storage. Thresholds promoted to float32 at comparison time. Balance of accuracy and cache efficiency.

# **7\. Output Artifact Formats**

## **7.1 Shared Library (.so / .dll / .dylib)**

The primary output for server-side deployments. The compiled inference function is exposed through a stable C ABI:

int timber\_infer(

    const float\*  inputs,      // input feature vector, length \= n\_features

    int           n\_samples,   // number of samples in batch

    float\*        outputs,     // output buffer, pre-allocated by caller

    TimberCtx\*    ctx          // opaque context, zero-copy thread safe

);

The context object holds compiled model state (the node array, threshold array, leaf values) and is initialized once at process startup. It is read-only after initialization and safe to use concurrently from multiple threads without locking.

## **7.2 Static Library (.a / .lib)**

For embedding directly into a host application binary. Same C ABI as shared library. Useful for mobile applications, command-line tools, and environments where dynamic linking is unavailable or undesirable.

## **7.3 WebAssembly Module (.wasm)**

For browser-side inference, edge worker deployment, and sandboxed server environments. The WASM module exposes the same logical interface via a WASM-compatible calling convention. Input and output buffers are passed as linear memory offsets.

## **7.4 C Source Package**

A self-contained directory containing:

* model.c — the compiled inference logic

* model.h — the public header

* model\_data.c — the node array and threshold data as static const arrays

* CMakeLists.txt — build configuration for the target

* Makefile — fallback build for environments without CMake

* audit\_report.json — compilation report (see Section 9\)

This output is the primary format for embedded and safety-critical targets. It is human-readable, grep-able, diffable between model versions, and compilable with any conforming C99 toolchain.

## **7.5 Python Wheel (timber-runtime)**

A thin Python package containing only the compiled shared library and a ctypes wrapper. No scikit-learn, no XGBoost, no pandas dependency. Import and call:

from timber\_runtime import TimberModel

model \= TimberModel.load('model.timber')

predictions \= model.predict(X\_numpy)  \# X is a numpy float32 array

This is the migration path for teams who want the performance benefits of compiled inference without changing their serving infrastructure.

# **8\. Target Hardware Specifications**

## **8.1 Cloud / Server Targets (Phase 1\)**

| Target | SIMD Width | Key Features |
| :---- | :---- | :---- |
| x86-64 generic | SSE2 (128-bit) | Baseline, broadly compatible |
| x86-64 AVX2 | AVX2 (256-bit) | 8-wide float32 vectorization |
| x86-64 AVX-512 | AVX-512 (512-bit) | 16-wide float32 vectorization, masking, preferred target |
| ARM64 (Graviton2/3) | NEON (128-bit) | 4-wide float32, AWS Graviton optimization |
| ARM64 with SVE | SVE (scalable) | Scalable vectorization for Ampere Altra, future Graviton |

## **8.2 Edge / Embedded Targets (Phase 2\)**

| Target | ISA | Notes |
| :---- | :---- | :---- |
| ARM Cortex-M4/M7 | Thumb-2 \+ FPU | IoT, wearables, industrial sensors — integer-only mode available |
| ARM Cortex-A55 | AArch64 \+ NEON | Mobile, automotive infotainment |
| RISC-V RV32GC | RISC-V 32-bit | Emerging embedded targets, open ISA |
| x86 (i686) | x86 SSE2 | Legacy industrial systems |

## **8.3 Target Specification File**

The target is specified as a TOML file passed to the compiler. Example:

\[target\]

arch \= "x86\_64"

features \= \["avx512f", "avx512bw", "avx512vl"\]

os \= "linux"

abi \= "systemv"

\[precision\]

mode \= "float32"

\[output\]

format \= "shared\_library"

strip\_symbols \= true

# **9\. Command-Line Interface**

The primary user interface is the timber CLI. It provides commands for compilation, inspection, benchmarking, and validation.

## **9.1 Compile**

timber compile \\

  \--model  model.json            \\   \# input model artifact

  \--format xgboost               \\   \# input format hint (auto-detected if omitted)

  \--target targets/avx512.toml   \\   \# hardware target spec

  \--out    ./dist/               \\   \# output directory

  \--calibration calib.csv        \\   \# optional: calibration data for freq-ordering

  \--pipeline pipeline.pkl        \\   \# optional: scikit-learn Pipeline for preprocessing

## **9.2 Inspect**

timber inspect model.json

Prints a summary of the model: framework, number of trees, maximum depth, number of features, objective, estimated compiled size. No compilation is performed.

## **9.3 Benchmark**

timber bench \\

  \--artifact  ./dist/model.so    \\

  \--data      bench\_data.csv     \\

  \--batch-sizes 1,16,64,256      \\

  \--warmup-iters 1000

Runs the compiled artifact against a data file and reports latency (P50/P95/P99), throughput (samples/second), and memory usage. Compares against a baseline (original framework) if \--baseline is provided.

## **9.4 Validate**

timber validate \\

  \--artifact  ./dist/model.so    \\

  \--reference model.json         \\

  \--data      validation.csv     \\

  \--tolerance 1e-5

Runs the compiled artifact and the reference framework side-by-side on a validation dataset. Reports maximum absolute error, mean absolute error, and any samples where the outputs diverge beyond the tolerance threshold.

# **10\. Audit Report**

Every compilation produces an audit\_report.json alongside the artifact. The report documents the complete compilation history and is intended to satisfy regulatory review requirements in financial services, healthcare, and automotive contexts.

## **10.1 Report Contents**

* Input artifact hash (SHA-256) — ties the compiled artifact to the exact model version

* Timber version and target spec — full reproducibility record

* Optimizer pass log — every pass that ran, what it changed, and what it left unchanged

* Pruning summary — how many leaves were pruned by dead leaf elimination, and the estimated accuracy impact

* Quantization decisions — which features were quantized, to what precision, and why

* Calibration data statistics — if calibration data was provided, summary statistics of the input distribution

* Output artifact hash (SHA-256) — the deterministic fingerprint of the compiled binary

*Determinism guarantee: given the same input artifact, target spec, and optimizer configuration, Timber will always produce the bit-identical output artifact. The audit report is reproducible.*

# **11\. Integration Paths**

## **11.1 Drop-in Python Replacement**

Teams migrating from scikit-learn or XGBoost prediction in Python can adopt Timber with minimal code changes by using the timber-runtime wheel:

\# Before

predictions \= model.predict(X)

\# After

from timber\_runtime import TimberModel

tmodel \= TimberModel.load('model.timber')

predictions \= tmodel.predict(X)  \# returns numpy array, same shape

The timber-runtime package has no dependency on the training framework. It imports in under 50ms and adds no startup penalty.

## **11.2 REST Endpoint (timber serve)**

For teams who want a ready-made HTTP inference endpoint without writing serving code:

timber serve \--artifact ./dist/model.so \--port 8080

This starts a minimal HTTP server (written in Rust, not Python) that accepts JSON or binary-encoded feature vectors and returns predictions. The server handles batching, concurrency, and health checks. It is not intended to replace a production API gateway but is sufficient for development and internal tooling.

## **11.3 Embedded / Bare Metal**

For embedded targets, the C source package output (Section 7.4) is the integration path. The generated code has the following guarantees:

* No dynamic memory allocation — all working memory is statically allocated at compile time

* No recursion — the traversal is iterative with a statically bounded stack

* No floating-point if target specifies integer-only mode

* No standard library dependencies beyond \<stdint.h\> and \<string.h\>

* Deterministic execution time — no data-dependent branching in the hot path (branch counts are fixed by the unrolled structure)

These properties make the output suitable for MISRA-C compliance review and IEC 61508 / ISO 26262 safety certification processes.

# **12\. Performance Targets**

The following targets represent the design goals for Phase 1\. They are validated by the benchmark suite in the repository.

| Scenario | Target | Baseline (Python) |
| :---- | :---- | :---- |
| XGBoost 500 trees, depth 6, batch=1, AVX-512 | \< 50 microseconds | \~2-5 milliseconds |
| XGBoost 500 trees, depth 6, batch=256, AVX-512 | \< 5ms total (\>50K samples/sec) | \~100-300ms |
| scikit-learn Pipeline (StandardScaler \+ RF 200 trees), batch=1 | \< 100 microseconds | \~5ms |
| LightGBM 1000 trees, depth 8, batch=1, AVX2 | \< 200 microseconds | \~8ms |
| XGBoost, batch=1, ARM Cortex-M7 (no SIMD) | \< 5ms | N/A (no Python) |

*Target latency improvements of 10-100x over the Python reference implementation. Memory footprint improvements of 50-200x (removing runtime dependencies).*

# **13\. Development Roadmap**

## **Phase 1 — Core Compiler (Months 1–6)**

* XGBoost and LightGBM front-end parsers

* Timber IR definition and serialization

* Optimizer passes 1–5 (dead leaf, constant feature, quantization, branch sorting, pipeline fusion)

* LLVM IR emitter for x86-64 (generic, AVX2, AVX-512)

* C99 emitter

* Shared library output format

* timber compile, inspect, validate CLI commands

* Benchmark suite with reference comparisons

* Audit report generation

## **Phase 2 — Expanded Support (Months 7–12)**

* scikit-learn Pipeline parser

* ONNX ML opset parser

* ARM64 NEON and SVE targets

* WebAssembly emitter

* timber serve HTTP endpoint

* timber-runtime Python wheel

* Vectorization pass (Pass 6\) for batched inference

* Embedded target profiles (Cortex-M, RISC-V)

## **Phase 3 — Enterprise Features (Months 13–18)**

* CatBoost and H2O MOJO parsers

* Stacking and voting ensemble support

* Multi-output model support (multi-class, multi-label, multi-target regression)

* Differential compilation — detect what changed between model versions and recompile only affected trees

* MISRA-C compliance mode for safety-critical C output

* GUI for audit report visualization

* Cloud SaaS: compile via API, serve via Timber-hosted endpoint

# **14\. Non-Goals**

To maintain focus, the following are explicitly out of scope for Timber:

* Training. Timber does not train models. It consumes artifacts from existing training frameworks.

* Neural network inference. Deep learning is a solved (and crowded) problem. Timber does not compete with TensorRT, ONNX Runtime's neural net path, or TVM for this workload.

* Model monitoring and drift detection. Timber is an inference compiler, not an MLOps platform.

* Feature engineering at training time. Timber may compile preprocessing transformers that were fitted at training time, but it does not perform feature engineering on raw data.

* GPU inference for classical ML. The tree traversal problem is not well-suited to GPUs for batch sizes typical of classical ML deployments. This may be revisited if customer demand warrants it.

*Timber — Classical ML Inference Compiler  |  Technical Product Documentation v0.1*