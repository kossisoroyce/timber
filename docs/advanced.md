# Advanced Usage

## Direct Compilation (No Store)

For embedding compiled models directly into C/C++ projects:

```bash
# Compile to C99 source
timber compile --model model.json --out ./dist/

# With a specific hardware target
timber compile --model model.json --target targets/x86_64_avx2.toml --out ./dist/

# With calibration data for branch sorting optimization
timber compile --model model.json --calibration-data train.csv --out ./dist/
```

This produces a self-contained directory:

```
dist/
├── model.c              # Inference logic
├── model.h              # Public API header
├── model_data.c         # Tree data as static const arrays
├── CMakeLists.txt       # CMake build config
├── Makefile             # GNU Make fallback
├── model.timber.json    # Serialized IR (for debugging)
└── audit_report.json    # Compilation audit trail
```

### Building

```bash
cd dist/
make                     # Produces libtimber_model.so and libtimber_model.a
```

Or with CMake:

```bash
cd dist/
mkdir build && cd build
cmake ..
make
```

## Target Hardware Specifications

Customize code generation with a TOML target file:

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

- `targets/x86_64_generic.toml` — Baseline SSE2
- `targets/x86_64_avx2.toml` — AVX2 + FMA
- `targets/x86_64_avx512.toml` — AVX-512
- `targets/arm64_neon.toml` — ARM64 NEON

## WebAssembly Output

For browser and edge deployment:

```python
from timber.codegen.wasm import WasmEmitter
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")
emitter = WasmEmitter(ir)
files = emitter.emit()

# files contains:
#   "model.wat"         — WebAssembly Text Format
#   "timber_model.js"   — JavaScript bindings
```

Usage in the browser:

```html
<script src="timber_model.js"></script>
<script>
  loadTimberModel("model.wat").then(model => {
    const prediction = model.predict([17.99, 10.38, /* ... */]);
    console.log("Prediction:", prediction);
  });
</script>
```

## MISRA-C Compliance Mode

For safety-critical deployments (automotive, medical, avionics):

```python
from timber.codegen.misra_c import MisraCEmitter, check_misra_compliance
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")

# Generate MISRA-C compliant code
emitter = MisraCEmitter(ir)
files = emitter.emit()

# Validate compliance
report = check_misra_compliance(files)
print(f"Violations: {report.violations}")
print(f"Warnings: {report.warnings}")
print(f"Compliant: {report.is_compliant}")
```

## Differential Compilation

When retraining models incrementally, avoid full recompilation:

```python
from timber.optimizer.diff_compile import diff_models, incremental_compile
from timber.frontends.auto_detect import parse_model

old_ir = parse_model("model_v1.json")
new_ir = parse_model("model_v2.json")

# Compute diff
diff = diff_models(old_ir, new_ir)
print(f"Added: {len(diff.added)} trees")
print(f"Removed: {len(diff.removed)} trees")
print(f"Modified: {len(diff.modified)} trees")
print(f"Unchanged: {len(diff.unchanged)} trees")

# Incremental compile (only recompiles changed trees)
updated_ir = incremental_compile(old_ir, new_ir)
```

## Ensemble Composition

Compose multiple models into voting or stacking ensembles:

```python
from timber.ir.ensemble_meta import VotingEnsembleStage, StackingEnsembleStage
from timber.frontends.auto_detect import parse_model

# Load base models
model_a = parse_model("xgb_model.json")
model_b = parse_model("lgb_model.txt")

# Voting ensemble (weighted average)
voting = VotingEnsembleStage(
    sub_models=[model_a, model_b],
    weights=[0.6, 0.4],
    voting="soft"
)

# Stacking ensemble (meta-learner)
meta_learner = parse_model("meta_model.json")
stacking = StackingEnsembleStage(
    base_models=[model_a, model_b],
    meta_model=meta_learner,
    passthrough=True  # Also pass original features to meta-learner
)
```

## Runtime Logging

Debug compiled models with the logging callback:

```c
#include "model.h"
#include <stdio.h>

void my_logger(int level, const char* msg) {
    const char* levels[] = {"ERROR", "WARN", "INFO", "DEBUG"};
    printf("[timber][%s] %s\n", levels[level], msg);
}

int main() {
    timber_set_log_callback(my_logger);

    TimberCtx* ctx;
    timber_init(&ctx);  // Logs: [timber][INFO] timber_init: OK

    // ... inference ...

    timber_free(ctx);
    return 0;
}
```

## Model Store Location

By default, models are stored in `~/.timber/models/`. Override with the `TIMBER_HOME` environment variable:

```bash
export TIMBER_HOME=/opt/timber
timber load model.json --name production-model
# Stored in /opt/timber/models/production-model/
```

## Audit Reports

Every compilation produces a JSON audit report:

```json
{
  "timber_version": "0.1.0",
  "timestamp": "2026-02-28T00:00:00Z",
  "input_hash": "sha256:abc123...",
  "model_summary": {
    "n_trees": 50,
    "n_features": 30,
    "objective": "binary:logistic"
  },
  "passes": [
    {"name": "dead_leaf_elimination", "changed": true, "duration_ms": 1.2},
    {"name": "constant_feature_detection", "changed": false, "duration_ms": 0.8}
  ],
  "output_files": {
    "model.c": "sha256:def456...",
    "model.h": "sha256:ghi789..."
  }
}
```

This supports regulatory compliance in finance (SOX, MiFID II) and healthcare (FDA, IEC 62304).
