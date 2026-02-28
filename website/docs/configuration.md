---
sidebar_position: 10
title: Configuration
---

# Configuration

Timber is configured via CLI flags, environment variables, and target specification files.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMBER_HOME` | `~/.timber` | Root directory for the model store |
| `CC` | `gcc` | C compiler for shared library compilation |

```bash
# Example: custom store location
export TIMBER_HOME=/opt/timber/models
timber load model.json --name prod-model

# Example: use clang
CC=clang timber load model.json --name my-model
```

## Model Store Structure

```
~/.timber/
├── models/
│   ├── my-model/
│   │   ├── model.c
│   │   ├── model.h
│   │   ├── model_data.c
│   │   ├── libtimber_model.so
│   │   ├── model.timber.json
│   │   ├── model_info.json
│   │   ├── audit_report.json
│   │   ├── CMakeLists.txt
│   │   └── Makefile
│   └── another-model/
│       └── ...
└── registry.json
```

The `registry.json` file indexes all cached models with their metadata.

## Target Specifications

For `timber compile`, you can provide a target spec TOML file:

```toml
# targets/x86_64_avx2.toml
[target]
arch = "x86_64"
features = ["avx2", "fma"]
os = "linux"
abi = "systemv"

[precision]
mode = "float32"

[output]
format = "c_source"
strip_symbols = false
```

Built-in targets:
- `x86_64_generic` — Baseline SSE2
- `x86_64_avx2` — AVX2 + FMA
- `arm64_neon` — ARM64 NEON

## Server Configuration

The built-in HTTP server is configured via `timber serve` flags:

```bash
timber serve my-model --host 0.0.0.0 --port 11434
```

For production deployments, front the built-in server with nginx or a load balancer.
