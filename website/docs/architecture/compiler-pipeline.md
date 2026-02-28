---
sidebar_position: 1
title: Compiler Pipeline
---

# Compiler Pipeline

Timber follows a classical compiler architecture with four phases.

## Overview

```
                    ┌─────────────┐
                    │ Model File  │
                    │ .json .pkl  │
                    │ .txt .onnx  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Front-End  │  5 format-specific parsers
                    │  (Parsing)  │  → Framework-agnostic IR
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Middle-End │  6 optimization passes
                    │ (Optimizer) │  → Optimized IR
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Back-End   │  3 code emitters
                    │  (Codegen)  │  → C99 / WASM / MISRA-C
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Native     │  gcc/clang compilation
                    │  Compiler   │  → .so / .dylib / .a
                    └─────────────┘
```

## Phase 1: Front-End

**Input:** Framework-specific model artifact
**Output:** `TimberIR` — a list of pipeline stages

Each parser converts its native format into Timber's IR. The IR uses a generic tree representation that abstracts away framework-specific details:

- **XGBoost:** Converts base_score from probability to logit space, handles `default_left` for missing values
- **LightGBM:** Handles negative-indexed leaves, re-indexes to 0-based
- **scikit-learn:** Traverses sklearn tree arrays, handles Pipeline with StandardScaler
- **CatBoost:** Expands oblivious (symmetric) trees into general form
- **ONNX:** Reconstructs trees from flat node arrays in TreeEnsemble operators

**Entry point:** `timber.frontends.auto_detect.parse_model()`

## Phase 2: Middle-End (Optimizer)

**Input:** `TimberIR`
**Output:** Optimized `TimberIR`

Six passes run sequentially, each transforming the IR. Passes are independent and can be skipped or reordered. Each pass produces an audit log entry.

See [Optimization Passes](/docs/architecture/optimization-passes) for details.

**Entry point:** `timber.optimizer.pipeline.run()`

## Phase 3: Back-End (Code Generation)

**Input:** Optimized `TimberIR`
**Output:** Dictionary of `{filename: content}` pairs

Three emitters are available:
- **C99** (`c99.py`) — primary target for servers and embedded
- **WebAssembly** (`wasm.py`) — browser and edge deployment
- **MISRA-C** (`misra_c.py`) — wraps C99 emitter with compliance transformations

**Entry point:** `timber.codegen.c99.C99Emitter.emit()`

## Phase 4: Native Compilation

**Input:** Generated C source files
**Output:** Shared library (`.so` / `.dylib`)

Uses the system's C compiler (`gcc` or `clang`) with `-O3 -shared -std=c99`. The `Makefile` and `CMakeLists.txt` generated alongside the source provide build configuration.

## Pipeline Orchestration

The `timber load` command orchestrates all four phases:

```python
# Simplified flow inside cli.py
ir = parse_model(model_path, format=format)      # Phase 1
ir = optimizer_pipeline.run(ir)                    # Phase 2
files = C99Emitter(ir).emit()                      # Phase 3
subprocess.run(["gcc", "-O3", "-shared", ...])     # Phase 4
```

The `ModelStore` then caches the compiled artifact in `~/.timber/models/<name>/`.
