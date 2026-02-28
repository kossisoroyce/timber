---
sidebar_position: 4
title: Embedding in C/C++
---

# Embedding in C/C++

Timber generates self-contained C99 source code that you can embed directly in any C/C++ project.

## Compile to Source

```bash
timber compile --model model.json --out ./inference/
```

This produces:

```
inference/
├── model.h          # Public API
├── model.c          # Inference logic
├── model_data.c     # Tree data (static const)
├── CMakeLists.txt
└── Makefile
```

## Using the Generated API

```c
#include "model.h"
#include <stdio.h>

int main() {
    // Initialize
    TimberCtx* ctx;
    int err = timber_init(&ctx);
    if (err != TIMBER_OK) {
        printf("Init failed: %s\n", timber_strerror(err));
        return 1;
    }

    // Single inference
    float inputs[TIMBER_N_FEATURES] = {17.99, 10.38, 122.8, /* ... */};
    float outputs[TIMBER_N_OUTPUTS];

    err = timber_infer_single(inputs, outputs, ctx);
    if (err != TIMBER_OK) {
        printf("Inference failed: %s\n", timber_strerror(err));
        return 1;
    }
    printf("Prediction: %f\n", outputs[0]);

    // Batch inference
    float batch[100 * TIMBER_N_FEATURES];
    float results[100 * TIMBER_N_OUTPUTS];
    // ... fill batch ...
    err = timber_infer(batch, 100, results, ctx);

    // Cleanup
    timber_free(ctx);
    return 0;
}
```

## Building

With Make:

```bash
cd inference/
make  # Produces libtimber_model.so and libtimber_model.a
```

With CMake:

```bash
cd inference/
mkdir build && cd build
cmake .. && make
```

Link against your project:

```bash
gcc -O2 -o my_app my_app.c -I./inference -L./inference -ltimber_model -lm
```

## Error Codes

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `TIMBER_OK` | Success |
| -1 | `TIMBER_ERR_NULL` | Null pointer argument |
| -2 | `TIMBER_ERR_INIT` | Context not initialized |
| -3 | `TIMBER_ERR_BOUNDS` | Argument out of bounds |

## Runtime Logging

Set a callback to receive log messages from the generated code:

```c
void my_logger(int level, const char* msg) {
    const char* labels[] = {"ERROR", "WARN", "INFO", "DEBUG"};
    fprintf(stderr, "[timber][%s] %s\n", labels[level], msg);
}

// Register before inference
timber_set_log_callback(my_logger);
```

When no callback is set, logging is a no-op with zero overhead.

## Constants

The generated header provides compile-time constants:

```c
#define TIMBER_N_FEATURES  30    // Input feature count
#define TIMBER_N_OUTPUTS   1     // Output count
#define TIMBER_N_TREES     50    // Tree count
#define TIMBER_ABI_VERSION 1     // ABI version for compatibility
```
