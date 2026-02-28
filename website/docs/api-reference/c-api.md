---
sidebar_position: 4
title: C API
---

# C API Reference

The generated C code exposes a clean, minimal API suitable for embedded systems, servers, and safety-critical applications.

## Header: `model.h`

```c
#include "model.h"
```

### Constants

```c
#define TIMBER_N_FEATURES  30    // Number of input features
#define TIMBER_N_OUTPUTS   1     // Number of outputs per sample
#define TIMBER_N_TREES     50    // Number of trees in ensemble
#define TIMBER_ABI_VERSION 1     // ABI version for compatibility checking
```

### Error Codes

```c
#define TIMBER_OK         0     // Success
#define TIMBER_ERR_NULL  -1     // Null pointer argument
#define TIMBER_ERR_INIT  -2     // Context not initialized
#define TIMBER_ERR_BOUNDS -3    // Argument out of bounds
```

### Types

```c
typedef struct TimberCtx TimberCtx;   // Opaque context handle
```

---

## Functions

### `timber_init`

Initialize a Timber context.

```c
int timber_init(TimberCtx** ctx);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `TimberCtx**` | Output: pointer to new context |
| **Returns** | `int` | `TIMBER_OK` on success, error code otherwise |

The context holds read-only state. After initialization, it is safe to share across threads.

### `timber_infer`

Batch inference.

```c
int timber_infer(const float* inputs, int n_samples, float* outputs, const TimberCtx* ctx);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `const float*` | Input array, shape `[n_samples × TIMBER_N_FEATURES]` |
| `n_samples` | `int` | Number of samples |
| `outputs` | `float*` | Output array, shape `[n_samples × TIMBER_N_OUTPUTS]` |
| `ctx` | `const TimberCtx*` | Initialized context |
| **Returns** | `int` | `TIMBER_OK` on success |

### `timber_infer_single`

Single-sample inference (convenience wrapper).

```c
int timber_infer_single(const float* inputs, float* outputs, const TimberCtx* ctx);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `const float*` | Input array, length `TIMBER_N_FEATURES` |
| `outputs` | `float*` | Output array, length `TIMBER_N_OUTPUTS` |
| `ctx` | `const TimberCtx*` | Initialized context |
| **Returns** | `int` | `TIMBER_OK` on success |

### `timber_free`

Free a Timber context.

```c
void timber_free(TimberCtx* ctx);
```

### `timber_strerror`

Convert error code to human-readable string.

```c
const char* timber_strerror(int err);
```

Example:
```c
int err = timber_init(&ctx);
if (err != TIMBER_OK) {
    fprintf(stderr, "Error: %s\n", timber_strerror(err));
}
```

### `timber_set_log_callback`

Register a logging callback. When not set, logging is a no-op.

```c
typedef void (*timber_log_fn)(int level, const char* message);
void timber_set_log_callback(timber_log_fn callback);
```

Log levels: `0` = ERROR, `1` = WARN, `2` = INFO, `3` = DEBUG.

---

## Complete Example

```c
#include "model.h"
#include <stdio.h>
#include <stdlib.h>

void logger(int level, const char* msg) {
    const char* labels[] = {"ERROR", "WARN", "INFO", "DEBUG"};
    fprintf(stderr, "[timber][%s] %s\n", labels[level], msg);
}

int main() {
    timber_set_log_callback(logger);

    TimberCtx* ctx;
    int err = timber_init(&ctx);
    if (err != TIMBER_OK) {
        fprintf(stderr, "Init failed: %s\n", timber_strerror(err));
        return 1;
    }

    // Single prediction
    float inputs[TIMBER_N_FEATURES] = {17.99, 10.38, 122.8, /* ... */};
    float outputs[TIMBER_N_OUTPUTS];
    err = timber_infer_single(inputs, outputs, ctx);
    printf("Prediction: %f\n", outputs[0]);

    // Batch prediction
    int n = 1000;
    float* batch_in = malloc(n * TIMBER_N_FEATURES * sizeof(float));
    float* batch_out = malloc(n * TIMBER_N_OUTPUTS * sizeof(float));
    // ... fill batch_in ...
    err = timber_infer(batch_in, n, batch_out, ctx);
    printf("Batch done: %d samples\n", n);

    free(batch_in);
    free(batch_out);
    timber_free(ctx);
    return 0;
}
```

## Building

```bash
# Shared library
gcc -O3 -shared -fPIC -std=c99 -o libtimber_model.so model.c model_data.c -lm

# Static library
gcc -O3 -c -std=c99 model.c model_data.c -lm
ar rcs libtimber_model.a model.o model_data.o

# Link with your application
gcc -O2 -o my_app my_app.c -L. -ltimber_model -lm
```

## Thread Safety

The `TimberCtx` is **read-only after `timber_init()`**. Multiple threads can call `timber_infer()` or `timber_infer_single()` concurrently with the same context without synchronization.
