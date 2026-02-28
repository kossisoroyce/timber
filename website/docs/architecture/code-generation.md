---
sidebar_position: 4
title: Code Generation
---

# Code Generation

The code generation phase transforms the optimized IR into self-contained source files.

## C99 Emitter

**File:** `timber/codegen/c99.py`

The primary emitter produces five files:

### `model.h` — Public Header

```c
#ifndef TIMBER_MODEL_H
#define TIMBER_MODEL_H

#define TIMBER_N_FEATURES  30
#define TIMBER_N_OUTPUTS   1
#define TIMBER_N_TREES     50
#define TIMBER_ABI_VERSION 1

#define TIMBER_OK          0
#define TIMBER_ERR_NULL   -1
#define TIMBER_ERR_INIT   -2
#define TIMBER_ERR_BOUNDS -3

typedef struct TimberCtx TimberCtx;
typedef void (*timber_log_fn)(int level, const char* message);

int timber_init(TimberCtx** ctx);
int timber_infer(const float* inputs, int n_samples, float* outputs, const TimberCtx* ctx);
int timber_infer_single(const float* inputs, float* outputs, const TimberCtx* ctx);
void timber_free(TimberCtx* ctx);
const char* timber_strerror(int err);
void timber_set_log_callback(timber_log_fn callback);

#endif
```

### `model_data.c` — Tree Data

All tree data is stored as `static const` arrays:

```c
static const int feature_indices_tree_0[] = {27, 22, 7, -1, -1, ...};
static const float thresholds_tree_0[] = {0.0912f, 105.95f, ...};
static const int left_children_tree_0[] = {1, 3, 5, -1, -1, ...};
static const int right_children_tree_0[] = {2, 4, 6, -1, -1, ...};
static const float leaf_values_tree_0[] = {0.0f, 0.0f, ..., 0.247f, ...};
```

This approach means:
- **Zero dynamic allocation** — all data lives in the `.rodata` section
- **Cache-friendly** — contiguous memory layout
- **No deserialization** — data is ready to use at load time

### `model.c` — Inference Logic

The inference function uses iterative tree traversal:

```c
static inline float traverse_tree(
    const float* inputs,
    const int* features, const float* thresholds,
    const int* left, const int* right,
    const float* leaves, int n_nodes
) {
    int node = 0;
    while (features[node] >= 0) {  // Not a leaf
        float val = inputs[features[node]];
        // NaN → default_left path
        if (val != val || val <= thresholds[node]) {
            node = left[node];
        } else {
            node = right[node];
        }
    }
    return leaves[node];
}
```

Key properties:
- **Iterative** — no recursion, bounded loop count (max = tree depth)
- **NaN handling** — `val != val` check sends NaN down default_left
- **Double accumulation** — sums in `double`, casts to `float` at the end

### Activation Functions

The emitter generates the appropriate activation:

- **Binary classification:** `sigmoid(x) = 1.0 / (1.0 + exp(-x))`
- **Multi-class:** softmax over per-class accumulators
- **Regression:** identity (raw sum)

## WebAssembly Emitter

**File:** `timber/codegen/wasm.py`

Produces WAT (WebAssembly Text Format) with the same tree traversal logic. Tree data is packed into WASM linear memory. The emitter generates a JavaScript loader that instantiates the module and exposes a `predict()` function.

## MISRA-C Emitter

**File:** `timber/codegen/misra_c.py`

Wraps the C99 emitter and applies MISRA-C:2012 compliance transformations:
- Unsigned integer literal suffixes (`42U`)
- No compiler-specific extensions
- Standard identifier protection
- Compliance declaration in headers
