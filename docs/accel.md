# `timber.accel` — Hardware Acceleration, Safety, Deployment

`timber.accel` is a built-in subsystem that extends the core Timber compiler with
SIMD / GPU / HLS / embedded backends, worst-case execution-time (WCET) analysis,
certification reporting (DO-178C, ISO 26262, IEC 62304), supply-chain primitives
(Ed25519 signing, AES-256-GCM encryption, TPM hooks), and deployment generators
(air-gapped bundles, C++ gRPC servers, ROS 2 nodes, PX4 modules).

Every `pip install timber-compiler` installs this module and the `timber-accel`
CLI — there is no separate package.

---

## CLI overview

```text
timber-accel compile      Compile to a SIMD / GPU / HLS / embedded target
timber-accel wcet         Worst-case execution-time analysis
timber-accel certify      Generate a certification report
timber-accel sign         Ed25519-sign an artifact
timber-accel verify       Verify an Ed25519 signature
timber-accel encrypt      AES-256-GCM encrypt an artifact
timber-accel decrypt      AES-256-GCM decrypt an artifact
timber-accel bundle       Create an air-gapped deployment tar.gz
timber-accel serve-native Generate a C++ gRPC / HTTP inference server
```

Run `timber-accel <cmd> --help` for the full option list of each subcommand.

---

## Target profiles

Target profiles are TOML files shipped under `timber/accel/targets/`. Reference
one by name (e.g. `x86_64_avx2_simd`) or pass a path to a custom `.toml`.

### SIMD

| Profile | ISA | Vector width | Notes |
|---------|-----|--------------|-------|
| `x86_64_avx2_simd` | AVX2 | 256 bits | FMA3 assumed |
| `x86_64_avx512_simd` | AVX-512 (F / DQ / BW) | 512 bits | Skylake-X and later |
| `arm_neon_simd` | ARM NEON | 128 bits | AArch64 baseline |
| `arm_sve_simd` | ARM SVE | scalable (VLA) | Graviton3, Neoverse-V1 |
| `riscv_rvv_simd` | RISC-V V 1.0 | scalable (VLA) | VLEN ≥ 128 recommended |

### GPU

| Profile | Backend | Targets |
|---------|---------|---------|
| `cuda_sm75` | CUDA 12 | Turing (T4, RTX 20xx) |
| `cuda_sm86` | CUDA 12 | Ampere (A10, RTX 30xx) |
| `metal_apple_m1` | Metal 3 | Apple Silicon M-series |
| `opencl_generic` | OpenCL 2.x | Portable fallback |

### FPGA HLS

| Profile | Tool | Notes |
|---------|------|-------|
| `hls_xilinx` | Vitis HLS | AXI4-Stream interface, pipelined traversal |
| `hls_intel`  | Intel FPGA SDK | OpenCL-style kernel |

### Embedded (no-heap, static buffers)

| Profile | Target | Cross-compile prefix |
|---------|--------|----------------------|
| `embedded_cortex_m4` | ARM Cortex-M4F | `arm-none-eabi-` |
| `embedded_esp32`     | Xtensa LX6 (ESP32) | `xtensa-esp32-elf-` |
| `embedded_stm32`     | Cortex-M7 (STM32H7) | `arm-none-eabi-` |

List all built-in targets:

```python
from timber.accel._util.target_loader import list_builtin_targets
print(list_builtin_targets())
```

---

## Example: AVX2 compile + sign

```bash
timber-accel compile \
    --model fraud.pkl \
    --target x86_64_avx2_simd \
    --deterministic \
    --sign \
    --out ./dist

ls ./dist
# model.c   model.h   model_data.c   CMakeLists.txt   Makefile
# ./dist.sig   timber_accel.pub
```

The SIMD emitter:

1. Runs the core Timber optimizer pipeline.
2. Emits baseline C99 via `C99Emitter`.
3. Expands `struct TimberCtx` with flat-array pointers needed for vectorised
   traversal.
4. Appends AVX2-intrinsic `timber_infer_simd()` to `model.c` and rewires
   `timber_infer_single` / `timber_infer` to delegate to it.
5. Patches `Makefile` / `CMakeLists.txt` with `-mavx2 -mfma` flags.

---

## WCET analysis

```bash
timber-accel wcet \
    --model anomaly.pkl \
    --arch cortex-m4 \
    --clock-mhz 168 \
    --safety-margin 3.0
```

Sample output (Cortex-M4 @ 168 MHz):

```text
WCET Analysis — cortex-m4 @ 168.0 MHz (safety margin: 3.0x)
  Raw cycles (worst): 4,218
  Cycles (worst):     12,654  (with 3.0x margin)
  Time (worst):       75.32 µs
  Raw cycles (avg):   2,810
  Cycles (avg):       8,430   (with 3.0x margin)
  Time (avg):         50.18 µs
    [trees] 8,230 cycles
    [logistic] 4,424 cycles
```

### Supported architectures

`cortex-m4`, `cortex-m7`, `x86_64`, `aarch64`, `riscv64`.

**Advisory notice** — the analytical model does not account for cache, branch
misprediction, or pipeline stalls. Real worst-case can be 3–10× higher. For
certified bounds use hardware-in-the-loop tools such as aiT, RapiTime, or
Bound-T.

---

## Certification reports

```bash
timber-accel certify \
    --model model.pkl \
    --profile do_178c \
    --include-wcet \
    --output cert.json
```

Supported profiles:

| Profile | Standard | Levels covered |
|---------|----------|----------------|
| `do_178c`   | RTCA DO-178C (airborne)     | A, B, C, D |
| `iso_26262` | ISO 26262 (automotive)      | ASIL A–D |
| `iec_62304` | IEC 62304 (medical)         | Class A, B, C |

The report is structured JSON containing:

- Source model metadata and hash
- Enabled optimizer passes and transforms
- MISRA-C compliance summary (when MISRA mode was used at compile time)
- Standard-specific heuristic checks (defensive programming, diagnostic
  coverage, unit verification markers, traceability of stage types)
- Optional embedded WCET block for a declared architecture

**Advisory notice** — checks are pattern-based and explicitly documented as
advisory. They are **not** a substitute for certified tools such as LDRA,
Polyspace, Astrée, or VectorCAST, nor for independent review by a qualified DER.

---

## Supply chain

### Ed25519 signing

```bash
# Generate a fresh keypair and sign the artifact directory
timber-accel sign --model ./dist --generate-key

# Verify with the public key
timber-accel verify --model ./dist --sig ./dist.sig --key timber_accel.pub
```

### AES-256-GCM encryption

```bash
# 256-bit hex key (or a file containing one)
timber-accel encrypt --model ./dist --key $TIMBER_KEY --output dist.enc
timber-accel decrypt --model dist.enc --key $TIMBER_KEY --output ./dist
```

### TPM integration

`timber.accel.safety.supply_chain.tpm` exposes hooks for Linux TPM 2.0 (via
`tpm2-pytss`) and a software emulator. Use these to attest-bind keys for model
provenance.

### Air-gapped deployment bundle

```bash
timber-accel bundle \
    --model model.pkl \
    --include-source \
    --include-cert \
    --output deploy.tar.gz
```

The tar.gz contains the compiled artifact, optional source `.c` files, optional
certification report, a `manifest.json` with hashes, and a `signatures/`
directory populated when `--sign` was used upstream.

---

## Deployment generators

### C++ gRPC server

```bash
timber-accel serve-native --model model.pkl --grpc --port 50051
# → serve_native/ with CMakeLists.txt, server.cc, proto/, client_example.cc
```

### ROS 2 node package

Generates a catkin-style package with a `rclpy` node that subscribes to a
`Float32MultiArray` topic, invokes the compiled model, and publishes the
output. Launch file included.

### PX4 module skeleton

Generates a PX4-style uORB module with `Run()` and `print_status()` entry
points wired to the compiled inference artifact, ready to drop into a PX4
firmware tree.

---

## Python API

Every CLI command has a programmatic equivalent:

```python
from timber.frontends import parse_model
from timber.accel._util.target_loader import load_target_profile
from timber.accel.accel.simd.base import get_simd_emitter
from timber.accel.safety.realtime.wcet import analyze_wcet
from timber.accel.safety.certification.report import generate_certification_report
from timber.accel.safety.supply_chain.signing import sign_artifact, generate_keypair

ir = parse_model("model.pkl")
profile = load_target_profile("x86_64_avx2_simd")
emitter = get_simd_emitter(profile)
output = emitter.emit(ir)
output.write("./dist")

wcet = analyze_wcet(ir, arch="cortex-m4", clock_mhz=168)
print(wcet["total_time_us_worst"])

report = generate_certification_report(ir, "do_178c", include_wcet=True)
open("cert.json", "w").write(report.to_json())

priv, pub = generate_keypair()
sign_artifact("./dist", key_path=priv)
```

---

## Where things live

```text
timber/accel/
├── accel/
│   ├── simd/        avx2.py  avx512.py  neon.py  sve.py  rvv.py  base.py
│   ├── gpu/         cuda.py  metal.py  opencl.py  base.py
│   ├── hls/         xilinx.py  intel.py  base.py
│   └── embedded/    cortex_m.py  esp32.py  stm32.py  base.py
├── safety/
│   ├── realtime/    wcet.py  deterministic.py  constant_time.py
│   ├── certification/ profiles.py  report.py  do_178c.py  iso_26262.py  iec_62304.py
│   └── supply_chain/ signing.py  verification.py  encryption.py  tpm.py
├── deploy/
│   ├── bundle/      bundler.py
│   ├── serve_native/ server_gen.py  grpc_proto.py
│   ├── autonomy/    ros2.py  px4.py
│   └── sensors/     radar.py  rf.py  telemetry.py
├── _util/           target_loader.py  crypto.py
├── cli.py           `timber-accel` entry point
├── targets/         18 built-in TOML target profiles
└── compliance_profiles/  DO-178C, ISO 26262, IEC 62304 TOML profiles
```
