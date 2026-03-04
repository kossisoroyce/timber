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

## LLVM IR Backend

Emit LLVM IR (`.ll`) for hardware-specific optimization or integration with existing LLVM toolchains:

```python
from timber.codegen.llvm_ir import LLVMIREmitter
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")

# Emit for the host architecture
emitter = LLVMIREmitter(target="x86_64")
out = emitter.emit(ir)
print(out.model_ll[:500])  # SSA-form LLVM IR text

# Save to disk
files = out.save("./dist/")
# files["model.ll"] — LLVM IR text file
```

Supported target triples:

| Alias | Triple emitted |
|-------|---------------|
| `x86_64` | `x86_64-unknown-linux-gnu` |
| `aarch64` | `aarch64-unknown-linux-gnu` |
| `cortex-m4` | `thumbv7em-none-eabi` |
| `cortex-m33` | `thumbv8m.main-none-eabi` |
| `rv32imf` | `riscv32-unknown-elf` |
| `rv64gc` | `riscv64-unknown-elf` |

Compile to native code with LLVM:

```bash
llc -filetype=obj model.ll -o model.o
clang model.o -shared -o model.so -lm
```

---

## Embedded Cross-Compilation

Target ARM Cortex-M and RISC-V microcontrollers with the built-in embedded profiles:

```python
from timber.codegen.c99 import C99Emitter, TargetSpec
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")

# Select an embedded profile
spec = TargetSpec.for_embedded("cortex-m4")
out = C99Emitter(spec).emit(ir)
out.write("./dist/")
```

The emitted `Makefile` automatically uses the correct cross-compiler:

| Profile | Toolchain | Flags |
|---------|-----------|-------|
| `cortex-m4` | `arm-none-eabi-gcc` | `-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard` |
| `cortex-m33` | `arm-none-eabi-gcc` | `-mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard` |
| `rv32imf` | `riscv32-unknown-elf-gcc` | `-march=rv32imf -mabi=ilp32f` |
| `rv64gc` | `riscv64-unknown-elf-gcc` | `-march=rv64gc -mabi=lp64d` |

Embedded builds produce a static `.a` library (no `-fPIC`, no `-shared`) suitable for bare-metal linking.

```bash
cd dist/
make            # invokes arm-none-eabi-gcc, produces libtimber_model.a
```

---

## MISRA-C Compliance Mode

For safety-critical deployments (automotive, medical, avionics):

```python
from timber.codegen.misra_c import MisraCEmitter
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")

# Generate MISRA-C:2012 compliant code
emitter = MisraCEmitter()
out = emitter.emit(ir)

# Check compliance (returns ComplianceReport)
report = emitter.check_compliance(out.model_c)
print(f"Compliant: {report.is_compliant}")
print(f"Violations: {len(report.violations)}")
print(f"Rules checked: {report.rules_checked}")
for v in report.violation_objects:
    print(f"  [{v.severity}] Rule {v.rule}: {v.description}")

# Write to disk and compile normally
out.write("./dist/")
```

Rules checked by the built-in verifier:

| Rule | Description |
|------|-------------|
| 1.1 | No compiler extensions (`__attribute__`, `__declspec`) |
| 7.1 | No octal integer literals |
| 14.4 | No VLAs |
| 20.4 | No `#undef` |
| 20.9 | No `<stdio.h>` include |
| 21.1 | No reserved identifier redefinition |
| 21.6 | No `printf`/`scanf` |
| 22.x | All variables initialized at declaration |

## Differential Privacy

Add calibrated noise to model outputs for privacy-preserving inference:

```python
from timber.privacy.dp import DPConfig, apply_dp_noise, calibrate_epsilon
import numpy as np

# Configure the mechanism
cfg = DPConfig(
    mechanism="laplace",   # or "gaussian"
    epsilon=1.0,           # privacy budget
    sensitivity=1.0,       # L1 sensitivity of your model output
    clip_outputs=True,
    output_min=0.0,
    output_max=1.0,
)

# Apply noise to raw inference outputs
raw_outputs = np.array([[0.85, 0.15], [0.32, 0.68]], dtype=np.float32)
noisy_outputs, report = apply_dp_noise(raw_outputs, cfg)

print(f"Mechanism:    {report.mechanism}")
print(f"Noise scale:  {report.noise_scale:.4f}")
print(f"Outputs:      {report.n_outputs_noised}")
print(report.summary())
```

**Mechanisms:**

| Mechanism | Noise scale | Best for |
|-----------|-------------|----------|
| `laplace` | `sensitivity / epsilon` | Unbounded outputs, `delta = 0` |
| `gaussian` | `√(2 ln(1.25/δ)) · sensitivity / epsilon` | Bounded outputs, (`ε`, `δ`)-DP |

**Calibrating epsilon** — find the privacy budget needed to limit noise to a target level:

```python
epsilon = calibrate_epsilon(
    noise_level=0.05,   # tolerable noise standard deviation
    sensitivity=1.0,
    mechanism="laplace",
)
print(f"Required epsilon: {epsilon:.3f}")
```

**Notes:**
- Input dtype is preserved exactly (float32 in → float32 out; float64 in → float64 out)
- Pass `seed=42` for deterministic, reproducible noise (useful for testing)
- Apply *after* C99 / Python inference, *before* returning results to clients

---

## Differential Compilation

When retraining models incrementally, avoid full recompilation:

```python
from timber.optimizer.diff_compile import diff_models, incremental_compile
from timber.frontends.auto_detect import parse_model

old_ir = parse_model("model_v1.json")
new_ir = parse_model("model_v2.json")

# Compute diff
diff = diff_models(old_ir, new_ir)
print(diff.summary())
# {"added": 3, "removed": 1, "modified": 2, "unchanged": 44}

# Incremental compile (reuses unchanged trees, annotates IR with diff metadata)
updated_ir = incremental_compile(old_ir, new_ir, diff)
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
