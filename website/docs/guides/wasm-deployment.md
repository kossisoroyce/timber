---
sidebar_position: 5
title: WebAssembly Deployment
---

# WebAssembly Deployment

Timber can emit WebAssembly Text Format (WAT) for browser and edge deployment.

## Generating WASM Output

```python
from timber.codegen.wasm import WasmEmitter
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")
emitter = WasmEmitter(ir)
files = emitter.emit()

# Write files
for filename, content in files.items():
    with open(filename, "w") as f:
        f.write(content)
```

This produces:

- `model.wat` — WebAssembly Text Format with tree traversal logic
- `timber_model.js` — JavaScript bindings with a simple `predict()` API

## Browser Usage

```html
<!DOCTYPE html>
<html>
<head><title>Timber WASM Inference</title></head>
<body>
  <script src="timber_model.js"></script>
  <script>
    loadTimberModel("model.wat").then(model => {
      const features = [17.99, 10.38, 122.8, /* ... 30 features */];
      const prediction = model.predict(features);
      console.log("Prediction:", prediction);
    });
  </script>
</body>
</html>
```

## Converting to Binary WASM

The emitter produces WAT (text format). Convert to binary `.wasm` with:

```bash
# Install wabt (WebAssembly Binary Toolkit)
brew install wabt    # macOS
apt install wabt     # Ubuntu

# Convert
wat2wasm model.wat -o model.wasm
```

## How It Works

The WASM emitter generates:

1. **Linear memory layout** — tree data (thresholds, feature indices, children, leaf values) packed into WASM linear memory
2. **`$traverse_tree` function** — iterative tree traversal matching the C99 emitter's logic
3. **`$timber_infer_single` export** — the main inference entry point
4. **Sigmoid activation** — for binary classification, implemented via `$exp_neg` helper

## Limitations

- Multi-class (softmax) is not yet supported in WASM
- Binary encoding requires external `wat2wasm` tool
- No SIMD instructions yet (future: wasm-simd proposal)
