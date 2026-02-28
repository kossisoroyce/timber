---
sidebar_position: 10
title: IoT & Edge Deployment
---

# IoT & Edge Deployment

Timber's compiled output is uniquely suited for IoT devices, microcontrollers, and edge gateways where Python is impractical or impossible.

## Why This Matters

The standard ML deployment path — Python + NumPy + framework runtime — is a non-starter on constrained hardware:

| Constraint | Typical IoT/Edge Device | Python ML Stack | Timber Output |
|------------|------------------------|-----------------|---------------|
| RAM | 256 KB – 64 MB | 50–200 MB | **< 100 KB** |
| Storage | 1–512 MB flash | 200+ MB | **48 KB .so** |
| OS | Bare-metal / RTOS / Linux | Full Linux only | **Any with C compiler** |
| CPU | ARM Cortex-M/A, RISC-V | x86/ARM + CPython | **Any C99 target** |
| Latency | < 1 ms required | 0.5–15 ms | **2 µs** |
| Power | Battery, solar | High (Python GC) | **Minimal** (no GC, no alloc) |
| Connectivity | Intermittent | Cloud API calls | **Fully offline** |

Timber eliminates the Python runtime entirely. The compiled artifact is a self-contained C99 shared library (or static library) with **zero dependencies**, **zero dynamic allocation**, and **zero recursion** — properties required by most embedded and safety-critical standards.

## Target Devices

### Microcontrollers (Cortex-M, RISC-V)

For bare-metal or RTOS deployments where there is no operating system or only a minimal one:

```bash
# Cross-compile for ARM Cortex-M4
timber compile --model model.json --out ./firmware/

# In your firmware build
arm-none-eabi-gcc -O2 -mcpu=cortex-m4 -mthumb \
  -c model.c model_data.c -I.
```

The generated code uses only:
- `static const` arrays (placed in flash/ROM)
- Stack-allocated locals (no heap)
- `<math.h>` for `exp()` (sigmoid only — can be replaced with a lookup table)

**Memory footprint for a 50-tree XGBoost model:**

| Section | Size |
|---------|------|
| `.rodata` (tree data) | ~35 KB |
| `.text` (inference code) | ~8 KB |
| Stack usage per call | ~200 bytes |
| **Total** | **~43 KB flash, ~200 B RAM** |

### Edge Gateways (Raspberry Pi, Jetson Nano, BeagleBone)

These have full Linux but limited resources. Timber runs natively:

```bash
# On the gateway itself
pip install timber-compiler
timber load model.json --name anomaly-detector
timber serve anomaly-detector --port 8080
```

Or cross-compile on your dev machine and deploy only the `.so`:

```bash
# On dev machine (cross-compile for aarch64)
timber compile --model model.json --out ./deploy/
aarch64-linux-gnu-gcc -O3 -shared -fPIC -std=c99 \
  -o libtimber_model.so model.c model_data.c -lm

# Copy to gateway
scp libtimber_model.so pi@gateway:~/models/
```

### WebAssembly for Edge Browsers/Kiosks

For browser-based edge inference (retail kiosks, industrial HMIs):

```python
from timber.codegen.wasm import WasmEmitter
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")
files = WasmEmitter(ir).emit()
# Deploy model.wat + timber_model.js to the edge device's browser
```

See [WebAssembly Deployment](/docs/guides/wasm-deployment) for details.

## Industry Use Cases

### Predictive Maintenance (Manufacturing)

Run vibration/temperature anomaly detection directly on the sensor node:

```c
// Runs on ARM Cortex-M4 @ 168 MHz
// Reads accelerometer, predicts failure probability
float features[TIMBER_N_FEATURES];
read_sensor_data(features);

float prediction[TIMBER_N_OUTPUTS];
timber_infer_single(features, prediction, ctx);

if (prediction[0] > 0.85f) {
    trigger_maintenance_alert();
}
```

**Benefits:** No cloud round-trip (saves 50–200 ms), works offline, runs on existing sensor hardware.

### Smart Agriculture

Soil moisture, pH, and weather data → irrigation decisions on a solar-powered gateway:

- Model: 30-tree GradientBoosting regressor
- Device: ESP32 or Raspberry Pi Zero
- Compiled artifact: 22 KB
- Inference: < 5 µs
- Power: runs on solar with weeks of battery backup

### Fleet/Vehicle Telematics

Driving behavior scoring or fault prediction in an OBD-II dongle:

- Input: CAN bus signals (RPM, speed, throttle, brake)
- Model: 100-tree XGBoost classifier
- Device: ARM Cortex-A53 (e.g., NXP i.MX6)
- Latency: < 10 µs per inference
- Certification: [MISRA-C compliant output](/docs/guides/misra-c-compliance) for automotive standards (ISO 26262)

### Retail / Point-of-Sale

Real-time fraud scoring at the payment terminal:

- Must respond within the payment authorization window (< 100 ms)
- Cannot rely on cloud connectivity (store may have intermittent internet)
- Timber gives **2 µs inference** — 50,000× faster than the time budget

### Medical Devices

Patient monitoring with on-device anomaly detection:

- Regulatory requirement: [IEC 62304](https://en.wikipedia.org/wiki/IEC_62304) software lifecycle
- Timber provides: MISRA-C code + deterministic [audit trails](/docs/guides/audit-trails)
- No dynamic allocation = no memory fragmentation over long-running operation

## Deployment Patterns

### Pattern 1: Compile Once, Deploy to Fleet

```
Dev Machine                         Edge Fleet
┌─────────────┐                    ┌─────────────┐
│ Train model  │                   │ Device A     │
│ timber load  │──── .so ─────────▶│ (ARM Cortex) │
│              │     48 KB         └─────────────┘
└─────────────┘         │          ┌─────────────┐
                        ├─────────▶│ Device B     │
                        │          │ (RISC-V)     │
                        │          └─────────────┘
                        │          ┌─────────────┐
                        └─────────▶│ Device C     │
                                   │ (x86 gateway)│
                                   └─────────────┘
```

Cross-compile for each target architecture, push the `.so` via OTA update. The artifact is **48 KB** — fits in any OTA budget.

### Pattern 2: On-Device Compilation

For Linux-based edge devices with a C compiler:

```bash
# On the device itself
timber load model.json --name local-model
```

### Pattern 3: WASM for Heterogeneous Fleet

When your fleet has mixed architectures and you can't cross-compile for each:

1. Emit WASM once
2. Deploy the same `.wasm` to every device with a WASM runtime (Wasmtime, WasmEdge, browser)
3. Near-native speed without per-platform compilation

## Comparison with Alternatives

| Approach | Artifact Size | Dependencies | Offline | Latency | MCU Support |
|----------|--------------|--------------|---------|---------|-------------|
| Python + XGBoost | 200+ MB | Python, NumPy, XGBoost | No | ~1 ms | No |
| ONNX Runtime | 50+ MB | ONNX Runtime C++ | Yes | ~100 µs | Limited |
| TFLite | 1+ MB | TFLite runtime | Yes | ~50 µs | Yes (with TFLite Micro) |
| **Timber** | **48 KB** | **None** | **Yes** | **2 µs** | **Yes** |

Timber's key advantage: **zero runtime dependencies**. The compiled code is self-contained C99 that works on any platform with a C compiler, from an 8-bit MCU to a cloud server.

## Getting Started on Edge

1. **Train your model** on your dev machine (any framework)
2. **Compile:**
   ```bash
   timber compile --model model.json --out ./edge-deploy/
   ```
3. **Cross-compile** for your target:
   ```bash
   arm-none-eabi-gcc -O2 -c model.c model_data.c
   arm-none-eabi-ar rcs libtimber_model.a model.o model_data.o
   ```
4. **Link** into your firmware/application
5. **Call** `timber_infer_single()` — 2 µs, zero allocation, fully offline

See [Embedding in C/C++](/docs/guides/embedding-in-c) for the full C API reference.
