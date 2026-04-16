---
sidebar_position: 11
title: Hardware Acceleration
---

# Hardware Acceleration & Safety

The `timber-accel` CLI provides hardware-accelerated compilation, real-time analysis, certification reporting, and secure deployment for safety-critical and high-performance inference.

## Installation

The acceleration module is included with `timber-compiler`. Install optional extras for GPU support:

```bash
# Base install (SIMD, embedded, safety, deployment)
pip install timber-compiler

# With GPU support (CUDA)
pip install timber-compiler[gpu]

# With ROS 2 support
pip install timber-compiler[ros]

# Full install
pip install timber-compiler[full]
```

## CLI Commands

### `timber-accel compile`

Compile to SIMD, GPU, HLS, or embedded targets.

```bash
timber-accel compile \
    --model fraud.pkl \
    --target x86_64_avx2_simd \
    --deterministic \
    --sign \
    --out ./dist
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | Required | Model file (sklearn, XGBoost, etc.) |
| `--target NAME` | `x86_64_generic` | Target profile: built-in name or TOML path |
| `--out DIR` | `./dist` | Output directory |
| `--deterministic` | False | Deterministic build (no timestamps) |
| `--sign` | False | Generate Ed25519 keypair and sign artifact |
| `--calibration-data PATH` | None | CSV for branch frequency optimization |

**Built-in targets:**

| Category | Profiles |
|----------|----------|
| SIMD | `x86_64_avx2_simd`, `x86_64_avx512_simd`, `arm_neon_simd`, `arm_sve_simd`, `riscv_rvv_simd` |
| GPU | `cuda_sm75`, `cuda_sm86`, `metal_apple_m1`, `opencl_generic` |
| FPGA HLS | `hls_xilinx`, `hls_intel` |
| Embedded | `embedded_cortex_m4`, `embedded_esp32`, `embedded_stm32` |

### `timber-accel wcet`

Worst-case execution time analysis for real-time systems.

```bash
timber-accel wcet \
    --model anomaly.pkl \
    --arch cortex-m4 \
    --clock-mhz 168 \
    --safety-margin 3.0
```

Output:
```
WCET Analysis — cortex-m4 @ 168.0 MHz (safety margin: 3.0x)
  Raw cycles (worst): 4,218
  Cycles (worst):     12,654  (with 3.0x margin)
  Time (worst):       75.32 µs
  Raw cycles (avg):   2,810
  Cycles (avg):       8,430   (with 3.0x margin)
  Time (avg):         50.18 µs
```

**Supported architectures:** `cortex-m4`, `cortex-m7`, `x86_64`, `aarch64`, `riscv64`

:::caution Advisory Notice
WCET estimates are analytical models that don't account for cache effects, branch misprediction, or pipeline stalls. Real worst-case may be 3–10× higher. For certified bounds, use hardware-in-the-loop tools (aiT, RapiTime, Bound-T).
:::

### `timber-accel certify`

Generate safety certification reports.

```bash
timber-accel certify \
    --model model.pkl \
    --profile do_178c \
    --include-wcet \
    --output cert.json
```

**Profiles:**

| Profile | Standard | Coverage |
|---------|----------|----------|
| `do_178c` | RTCA DO-178C (Aviation) | Levels A, B, C, D |
| `iso_26262` | ISO 26262 (Automotive) | ASIL A–D |
| `iec_62304` | IEC 62304 (Medical) | Class A, B, C |

The JSON report includes:
- Source model metadata and cryptographic hash
- Optimizer passes and transformations applied
- MISRA-C compliance summary (if applicable)
- Standard-specific heuristic checks
- Optional WCET block for declared architecture

:::caution Advisory Notice
Certification checks are pattern-based and advisory only. They are **not** a substitute for certified tools (LDRA, Polyspace, Astrée, VectorCAST) or independent review by a qualified DER/assessor.
:::

### `timber-accel sign` / `verify`

Ed25519 signing for supply chain integrity.

```bash
# Generate keypair and sign
timber-accel sign --model ./dist --generate-key

# Verify with public key
timber-accel verify --model ./dist --sig ./dist.sig --key timber_accel.pub
```

### `timber-accel encrypt` / `decrypt`

AES-256-GCM encryption for artifact protection.

```bash
# Encrypt
timber-accel encrypt --model ./dist --key $TIMBER_KEY --output dist.enc

# Decrypt
timber-accel decrypt --model dist.enc --key $TIMBER_KEY --output ./dist
```

### `timber-accel bundle`

Create air-gapped deployment packages.

```bash
timber-accel bundle \
    --model model.pkl \
    --target embedded_cortex_m4 \
    --include-source \
    --include-cert \
    --output deploy.tar.gz
```

Bundle contents:
- Compiled artifacts (`.c`, `.h`, `Makefile`, `CMakeLists.txt`)
- Optional source files (if `--include-source`)
- Optional certification report (if `--include-cert`)
- `manifest.json` with SHA-256 hashes
- `signatures/` directory (if `--sign` used)

### `timber-accel serve-native`

Generate C++ inference servers.

```bash
# gRPC server
timber-accel serve-native --model model.pkl --grpc --port 50051

# HTTP server
timber-accel serve-native --model model.pkl --http --port 8080

# Both
timber-accel serve-native --model model.pkl --grpc --http
```

Generated `serve_native/` directory contains:
- `server.cc` — gRPC/HTTP implementation
- `CMakeLists.txt` — build configuration
- `proto/` — Protocol Buffer definitions
- `client_example.cc` — sample client code

## Python API

All CLI commands have programmatic equivalents:

```python
from timber.frontends import parse_model
from timber.accel._util.target_loader import load_target_profile
from timber.accel.accel.simd.base import get_simd_emitter
from timber.accel.safety.realtime.wcet import analyze_wcet
from timber.accel.safety.certification.report import generate_certification_report
from timber.accel.safety.supply_chain.signing import sign_artifact, generate_keypair

# Parse and compile with SIMD
ir = parse_model("model.pkl")
profile = load_target_profile("x86_64_avx2_simd")
emitter = get_simd_emitter(profile)
output = emitter.emit(ir)
output.write("./dist")

# WCET analysis
wcet = analyze_wcet(ir, arch="cortex-m4", clock_mhz=168)
print(f"Worst-case: {wcet['total_time_us_worst']} µs")

# Certification report
report = generate_certification_report(ir, "do_178c", include_wcet=True)
open("cert.json", "w").write(report.to_json())

# Signing
priv, pub = generate_keypair()
sign_artifact("./dist", key_path=priv)
```

## Target Profile Reference

### SIMD Targets

| Profile | ISA | Width | Flags |
|---------|-----|-------|-------|
| `x86_64_avx2_simd` | AVX2 | 256-bit | `-mavx2 -mfma` |
| `x86_64_avx512_simd` | AVX-512 F/DQ/BW | 512-bit | `-mavx512f -mavx512dq -mavx512bw` |
| `arm_neon_simd` | ARM NEON | 128-bit | `-march=armv8-a+fp+simd` |
| `arm_sve_simd` | ARM SVE | Scalable | `-march=armv8-a+sve` |
| `riscv_rvv_simd` | RISC-V V 1.0 | Scalable | `-march=rv64gcv` |

### GPU Targets

| Profile | Backend | Minimum Version |
|---------|---------|-----------------|
| `cuda_sm75` | CUDA 12.0 | Turing (T4, RTX 20xx) |
| `cuda_sm86` | CUDA 12.0 | Ampere (A10, RTX 30xx) |
| `metal_apple_m1` | Metal 3 | macOS 13+ |
| `opencl_generic` | OpenCL 2.0 | Any OpenCL 2.x GPU |

### Embedded Targets

| Profile | MCU | Toolchain | Notes |
|---------|-----|-------------|-------|
| `embedded_cortex_m4` | Cortex-M4F | `arm-none-eabi-gcc` | No heap, static buffers |
| `embedded_cortex_m7` | Cortex-M7 | `arm-none-eabi-gcc` | Double precision FPU |
| `embedded_esp32` | ESP32 (Xtensa) | `xtensa-esp32-elf-gcc` | FreeRTOS compatible |
| `embedded_stm32` | STM32H7 | `arm-none-eabi-gcc` | CubeMX integration |

## Examples

### AVX2 Server Deployment

```bash
# Compile with AVX2
timber-accel compile \
    --model fraud_model.pkl \
    --target x86_64_avx2_simd \
    --sign \
    --out ./fraud_avx2

# Generate gRPC server
timber-accel serve-native \
    --model fraud_model.pkl \
    --grpc \
    --port 50051 \
    --out ./fraud_server

# Build and run
cd fraud_server && mkdir build && cd build
cmake .. && make -j
./timber_server --model ../../fraud_avx2 --port 50051
```

### Cortex-M4 Embedded

```bash
# Compile for embedded
timber-accel compile \
    --model sensor_model.pkl \
    --target embedded_cortex_m4 \
    --deterministic \
    --out ./sensor_fw

# Analyze WCET
timber-accel wcet \
    --model sensor_model.pkl \
    --arch cortex-m4 \
    --clock-mhz 168

# Create deployment bundle
timber-accel bundle \
    --model sensor_model.pkl \
    --target embedded_cortex_m4 \
    --include-source \
    --output sensor_v1.0.0.tar.gz
```

### Safety-Critical Aviation

```bash
# Compile with MISRA-C compliance
timber load model.json --name flight_controller --format sklearn

# Generate DO-178C certification report
timber-accel certify \
    --model model.json \
    --profile do_178c \
    --include-wcet \
    --arch cortex-m7 \
    --clock-mhz 480 \
    --output flight_controller_cert.json

# Sign for supply chain integrity
timber-accel sign \
    --model ~/.timber/models/flight_controller \
    --generate-key
```

## Architecture Overview

```
timber/accel/
├── accel/
│   ├── simd/           # AVX2, AVX-512, NEON, SVE, RISC-V V
│   ├── gpu/            # CUDA, Metal, OpenCL
│   ├── hls/            # Xilinx Vitis, Intel FPGA
│   └── embedded/       # Cortex-M4/M7, ESP32, STM32
├── safety/
│   ├── realtime/       # WCET, deterministic builds
│   ├── certification/  # DO-178C, ISO 26262, IEC 62304
│   └── supply_chain/   # Signing, encryption, TPM
├── deploy/
│   ├── bundle/         # Air-gapped deployment
│   ├── serve_native/   # C++ gRPC/HTTP servers
│   └── autonomy/       # ROS 2, PX4 integration
└── targets/            # 18 built-in TOML profiles
```

## Next Steps

- **[Examples](/docs/examples/xgboost)** — Framework-specific walkthroughs
- **[API Reference](/docs/api-reference/cli)** — Complete CLI documentation
- **[IoT Edge Deployment](/docs/guides/iot-edge-deployment)** — Embedded deployment patterns
