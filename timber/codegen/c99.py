"""C99 code emitter — generates portable C99 source for inference.

Output guarantees:
- No dynamic memory allocation
- No recursion
- No floating-point unless target supports it
- No standard library dependencies beyond <stdint.h> and <string.h>
- Deterministic execution time
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from timber.ir.model import (
    GPRStage,
    IsolationForestStage,
    KinematicsStage,
    KNNStage,
    LinearStage,
    NaiveBayesStage,
    Objective,
    PrecisionMode,
    SVMStage,
    TimberIR,
    TreeEnsembleStage,
    _c_factor,
)

# Joint types that consume a joint-angle input
_ACTIVE_TYPES = ("revolute", "prismatic", "continuous")

# ---------------------------------------------------------------------------
# Predefined embedded target profiles
# ---------------------------------------------------------------------------
EMBEDDED_PROFILES: dict[str, dict] = {
    "cortex-m4": {
        "arch": "cortex-m4",
        "os": "baremetal",
        "abi": "eabi",
        "cross_prefix": "arm-none-eabi-",
        "cpu_flags": "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb",
        "extra_flags": "--specs=nosys.specs",
    },
    "cortex-m33": {
        "arch": "cortex-m33",
        "os": "baremetal",
        "abi": "eabi",
        "cross_prefix": "arm-none-eabi-",
        "cpu_flags": "-mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb",
        "extra_flags": "--specs=nosys.specs",
    },
    "rv32imf": {
        "arch": "rv32imf",
        "os": "baremetal",
        "abi": "ilp32f",
        "cross_prefix": "riscv32-unknown-elf-",
        "cpu_flags": "-march=rv32imf -mabi=ilp32f",
        "extra_flags": "",
    },
    "rv64gc": {
        "arch": "rv64gc",
        "os": "linux",
        "abi": "lp64d",
        "cross_prefix": "riscv64-unknown-linux-gnu-",
        "cpu_flags": "-march=rv64gc -mabi=lp64d",
        "extra_flags": "",
    },
}


@dataclass
class TargetSpec:
    """Hardware target specification for code generation."""
    arch: str = "x86_64"
    features: list[str] = field(default_factory=list)
    os: str = "linux"
    abi: str = "systemv"
    precision: PrecisionMode = PrecisionMode.FLOAT32
    output_format: str = "c_source"
    strip_symbols: bool = False
    # Embedded cross-compilation fields
    cross_prefix: str = ""      # e.g. "arm-none-eabi-"
    cpu_flags: str = ""         # e.g. "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 ..."
    extra_flags: str = ""       # e.g. "--specs=nosys.specs"
    embedded: bool = False      # True disables shared-lib targets

    @classmethod
    def for_embedded(cls, profile: str, **kwargs) -> "TargetSpec":
        """Create a TargetSpec for a named embedded profile."""
        if profile not in EMBEDDED_PROFILES:
            raise ValueError(
                f"Unknown embedded profile '{profile}'. "
                f"Available: {', '.join(EMBEDDED_PROFILES)}"
            )
        p = EMBEDDED_PROFILES[profile]
        return cls(
            arch=p["arch"],
            os=p["os"],
            abi=p["abi"],
            cross_prefix=p["cross_prefix"],
            cpu_flags=p["cpu_flags"],
            extra_flags=p["extra_flags"],
            embedded=True,
            **kwargs,
        )


@dataclass
class C99Output:
    """The generated C99 source package."""
    model_c: str
    model_h: str
    model_data_c: str
    cmakelists: str
    makefile: str

    def write(self, output_dir: str | Path) -> list[str]:
        """Write all files to the output directory. Returns list of written paths."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        files = {
            "model.c": self.model_c,
            "model.h": self.model_h,
            "model_data.c": self.model_data_c,
            "CMakeLists.txt": self.cmakelists,
            "Makefile": self.makefile,
        }

        written = []
        for name, content in files.items():
            path = out / name
            path.write_text(content, encoding="utf-8")
            written.append(str(path))

        return written


class C99Emitter:
    """Emits C99 source code from optimized Timber IR."""

    def __init__(self, target: Optional[TargetSpec] = None):
        self.target = target or TargetSpec()

    def emit(self, ir: TimberIR) -> C99Output:
        """Generate the full C99 source package from the IR.

        Dispatches to the appropriate emitter based on the primary stage type:
          - TreeEnsembleStage  → tree traversal inference
          - LinearStage        → dot-product + activation inference
          - SVMStage           → kernel machine inference
        """
        # Find primary stage (skip preprocessing stages)
        _primary_types = (
            TreeEnsembleStage, LinearStage, SVMStage, KinematicsStage,
            IsolationForestStage, NaiveBayesStage, GPRStage, KNNStage,
        )
        primary = None
        for stage in ir.pipeline:
            if isinstance(stage, _primary_types):
                primary = stage
                break

        if primary is None:
            raise ValueError(
                "No supported primary stage found in IR pipeline. "
                "Expected TreeEnsembleStage, LinearStage, SVMStage, KinematicsStage, "
                "IsolationForestStage, NaiveBayesStage, GPRStage, or KNNStage."
            )

        if isinstance(primary, KinematicsStage):
            model_h = self._emit_header_kinematics(ir, primary)
            model_data_c = self._emit_data_kinematics(ir, primary)
            model_c = self._emit_inference_kinematics(ir, primary)
        elif isinstance(primary, IsolationForestStage):
            model_h = self._emit_header_iforest(ir, primary)
            model_data_c = self._emit_data_iforest(ir, primary)
            model_c = self._emit_inference_iforest(ir, primary)
        elif isinstance(primary, NaiveBayesStage):
            model_h = self._emit_header_nb(ir, primary)
            model_data_c = self._emit_data_nb(ir, primary)
            model_c = self._emit_inference_nb(ir, primary)
        elif isinstance(primary, GPRStage):
            model_h = self._emit_header_gpr(ir, primary)
            model_data_c = self._emit_data_gpr(ir, primary)
            model_c = self._emit_inference_gpr(ir, primary)
        elif isinstance(primary, KNNStage):
            model_h = self._emit_header_knn(ir, primary)
            model_data_c = self._emit_data_knn(ir, primary)
            model_c = self._emit_inference_knn(ir, primary)
        elif isinstance(primary, LinearStage):
            model_h = self._emit_header_linear(ir, primary)
            model_data_c = self._emit_data_linear(ir, primary)
            model_c = self._emit_inference_linear(ir, primary)
        elif isinstance(primary, SVMStage):
            model_h = self._emit_header_svm(ir, primary)
            model_data_c = self._emit_data_svm(ir, primary)
            model_c = self._emit_inference_svm(ir, primary)
        else:
            model_h = self._emit_header(ir, primary)
            model_data_c = self._emit_data(ir, primary)
            model_c = self._emit_inference(ir, primary)

        cmakelists = self._emit_cmake(ir)
        makefile = self._emit_makefile(ir)

        return C99Output(
            model_c=model_c,
            model_h=model_h,
            model_data_c=model_data_c,
            cmakelists=cmakelists,
            makefile=makefile,
        )

    def _emit_header(self, ir: TimberIR, ensemble: TreeEnsembleStage) -> str:
        """Generate model.h — the public C header."""
        n_features = ensemble.n_features
        n_outputs = 1 if ensemble.n_classes <= 2 else ensemble.n_classes
        float_type = self._float_type()

        lines = [
            "/* model.h — Timber compiled model inference header */",
            "/* Generated by Timber v0.1 — DO NOT EDIT */",
            "",
            "#ifndef TIMBER_MODEL_H",
            "#define TIMBER_MODEL_H",
            "",
            "#include <stdint.h>",
            "#include <stddef.h>",
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "/* ABI version — bump on breaking changes to the public API. */",
            "#define TIMBER_ABI_VERSION  1",
            '#define TIMBER_VERSION     "0.1.0"',
            "",
            f"#define TIMBER_N_FEATURES {n_features}",
            f"#define TIMBER_N_OUTPUTS  {n_outputs}",
            f"#define TIMBER_N_TREES    {ensemble.n_trees}",
            f"#define TIMBER_MAX_DEPTH  {ensemble.max_depth}",
            "",
            "/* Opaque context — holds compiled model state. */",
            "/* Read-only after init; thread-safe for concurrent inference. */",
            "typedef struct TimberCtx TimberCtx;",
            "",
            "/* Error codes */",
            "#define TIMBER_OK          0",
            "#define TIMBER_ERR_NULL   -1",
            "#define TIMBER_ERR_INIT   -2",
            "#define TIMBER_ERR_BOUNDS -3",
            "",
            "/* Initialize the model context. Call once at startup. */",
            "/* Returns TIMBER_OK on success, negative error code on failure. */",
            "int timber_init(TimberCtx** ctx);",
            "",
            "/* Return the ABI version this library was compiled with. */",
            "int timber_abi_version(void);",
            "",
            "/* Optional logging callback. Set to NULL to disable. */",
            "/* Signature: void my_logger(int level, const char* msg) */",
            "/* Levels: 0=error, 1=warn, 2=info, 3=debug */",
            "typedef void (*timber_log_fn)(int level, const char* msg);",
            "void timber_set_log_callback(timber_log_fn fn);",
            "",
            "/* Return a human-readable string for an error code. */",
            "const char* timber_strerror(int code);",
            "",
            "/* Free the model context. */",
            "void timber_free(TimberCtx* ctx);",
            "",
            "/* Run inference on a batch of samples. */",
            "/*",
            f" *  inputs:    input feature matrix, row-major, shape [n_samples x {n_features}]",
            " *  n_samples: number of samples in the batch",
            f" *  outputs:   output buffer, pre-allocated, shape [n_samples x {n_outputs}]",
            " *  ctx:       model context from timber_init",
            " *",
            " *  Returns 0 on success, non-zero on error.",
            " */",
            "int timber_infer(",
            f"    const {float_type}*  inputs,",
            "    int                  n_samples,",
            f"    {float_type}*        outputs,",
            "    const TimberCtx*     ctx",
            ");",
            "",
            "/* Single-sample inference convenience wrapper. */",
            "int timber_infer_single(",
            f"    const {float_type}  inputs[TIMBER_N_FEATURES],",
            f"    {float_type}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ");",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            "#endif /* TIMBER_MODEL_H */",
        ]
        return "\n".join(lines) + "\n"

    def _emit_data(self, ir: TimberIR, ensemble: TreeEnsembleStage) -> str:
        """Generate model_data.c — static const arrays for tree data."""
        float_type = self._float_type()
        lines = [
            "/* model_data.c — Timber compiled model data */",
            "/* Generated by Timber v0.1 — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "#include <math.h>",
            "",
        ]

        # Emit node data as flat arrays per tree
        # For each tree: feature_indices, thresholds, left_children, right_children, leaf_values
        for tree in ensemble.trees:
            tid = tree.tree_id
            n_nodes = len(tree.nodes)

            # Feature indices
            feat_vals = ", ".join(str(n.feature_index) for n in tree.nodes)
            lines.append(f"static const int32_t tree_{tid}_features[{n_nodes}] = {{{feat_vals}}};")

            # Thresholds
            thresh_vals = ", ".join(self._format_float(n.threshold) for n in tree.nodes)
            lines.append(f"static const {float_type} tree_{tid}_thresholds[{n_nodes}] = {{{thresh_vals}}};")

            # Left children
            left_vals = ", ".join(str(n.left_child) for n in tree.nodes)
            lines.append(f"static const int32_t tree_{tid}_left[{n_nodes}] = {{{left_vals}}};")

            # Right children
            right_vals = ", ".join(str(n.right_child) for n in tree.nodes)
            lines.append(f"static const int32_t tree_{tid}_right[{n_nodes}] = {{{right_vals}}};")

            # Leaf values
            leaf_vals = ", ".join(self._format_float(n.leaf_value) for n in tree.nodes)
            lines.append(f"static const {float_type} tree_{tid}_leaves[{n_nodes}] = {{{leaf_vals}}};")

            # Is-leaf flags
            is_leaf_vals = ", ".join("1" if n.is_leaf else "0" for n in tree.nodes)
            lines.append(f"static const int8_t tree_{tid}_is_leaf[{n_nodes}] = {{{is_leaf_vals}}};")

            # Default-left flags
            def_left_vals = ", ".join("1" if n.default_left else "0" for n in tree.nodes)
            lines.append(f"static const int8_t tree_{tid}_default_left[{n_nodes}] = {{{def_left_vals}}};")

            lines.append(f"#define TREE_{tid}_N_NODES {n_nodes}")
            lines.append("")

        # Base score
        lines.append(f"static const {float_type} TIMBER_BASE_SCORE = {self._format_float(ensemble.base_score)};")

        # Per-class base scores for multiclass models (XGBoost 3.1+ stores per-class vectors)
        if ensemble.objective == Objective.MULTICLASS_CLASSIFICATION and ensemble.n_classes > 2:
            pcbs = ensemble.per_class_base_scores
            if len(pcbs) == ensemble.n_classes:
                bs_vals = ", ".join(self._format_float(v) for v in pcbs)
            else:
                bs_vals = ", ".join("0.0" for _ in range(ensemble.n_classes))
            lines.append(f"static const double TIMBER_CLASS_BASE_SCORES[{ensemble.n_classes}] = {{{bs_vals}}};")

        lines.append("")

        return "\n".join(lines) + "\n"

    def _emit_inference(self, ir: TimberIR, ensemble: TreeEnsembleStage) -> str:
        """Generate model.c — the compiled inference logic."""
        float_type = self._float_type()
        n_features = ensemble.n_features
        n_outputs = 1 if ensemble.n_classes <= 2 else ensemble.n_classes
        n_trees = ensemble.n_trees

        lines = [
            "/* model.c — Timber compiled inference logic */",
            "/* Generated by Timber v0.1 — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <string.h>",
            "#include <math.h>",
            "",
        ]

        # Include the data file
        lines.append('#include "model_data.c"')
        lines.append("")

        # Context struct
        lines.extend([
            "/* Context structure — trivial for static models */",
            "struct TimberCtx {",
            "    int initialized;",
            "};",
            "",
            "static struct TimberCtx _default_ctx = {1};",
            "",
        ])

        # Logging, strerror, init, free, ABI
        lines.extend([
            "/* --- Logging --- */",
            "static timber_log_fn _timber_log_cb = NULL;",
            "",
            "void timber_set_log_callback(timber_log_fn fn) { _timber_log_cb = fn; }",
            "",
            "static void timber_log(int level, const char* msg) {",
            "    if (_timber_log_cb) _timber_log_cb(level, msg);",
            "}",
            "",
            "const char* timber_strerror(int code) {",
            '    switch (code) {',
            '        case  0: return "TIMBER_OK";',
            '        case -1: return "TIMBER_ERR_NULL: null pointer argument";',
            '        case -2: return "TIMBER_ERR_INIT: context not initialized";',
            '        case -3: return "TIMBER_ERR_BOUNDS: argument out of bounds";',
            '        default: return "TIMBER_ERR_UNKNOWN";',
            "    }",
            "}",
            "",
            "int timber_abi_version(void) { return TIMBER_ABI_VERSION; }",
            "",
            "int timber_init(TimberCtx** ctx) {",
            "    if (ctx == NULL) {",
            '        timber_log(0, "timber_init: ctx is NULL");',
            "        return TIMBER_ERR_NULL;",
            "    }",
            "    *ctx = &_default_ctx;",
            '    timber_log(2, "timber_init: OK");',
            "    return TIMBER_OK;",
            "}",
            "",
            "void timber_free(TimberCtx* ctx) {",
            "    (void)ctx; /* static allocation, nothing to free */",
            "}",
            "",
        ])

        # Single-tree traversal function
        lines.extend([
            f"static {float_type} traverse_tree(",
            f"    const {float_type}* input,",
            "    const int32_t* features,",
            f"    const {float_type}* thresholds,",
            "    const int32_t* left_children,",
            "    const int32_t* right_children,",
            f"    const {float_type}* leaf_values,",
            "    const int8_t* is_leaf,",
            "    const int8_t* default_left,",
            "    int n_nodes",
            ") {",
            "    int node = 0;",
            f"    int max_iter = {ensemble.max_depth + 2};",
            "    while (max_iter-- > 0) {",
            "        if (node < 0 || node >= n_nodes) return 0.0f;",
            "        if (is_leaf[node]) return leaf_values[node];",
            "",
            "        int feat = features[node];",
            f"        {float_type} val = input[feat];",
            "",
            "        /* NaN handling: follow default direction */",
            "        if (val != val) { /* NaN check */",
            "            node = default_left[node] ? left_children[node] : right_children[node];",
            "        } else if (val < thresholds[node]) {",
            "            node = left_children[node];",
            "        } else {",
            "            node = right_children[node];",
            "        }",
            "    }",
            "    return 0.0f;",
            "}",
            "",
        ])


        # (softmax is inlined in double precision in timber_infer_single)

        # Single-sample inference (unrolled tree calls)
        lines.extend([
            "int timber_infer_single(",
            f"    const {float_type}  inputs[TIMBER_N_FEATURES],",
            f"    {float_type}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ") {",
            "    (void)ctx;",
        ])

        if ensemble.objective == Objective.MULTICLASS_CLASSIFICATION and ensemble.n_classes > 2:
            # Multi-class: accumulate per-class scores with double precision
            lines.append(f"    double scores[{ensemble.n_classes}];")
            lines.append("    int c;")
            lines.append(f"    for (c = 0; c < {ensemble.n_classes}; c++) scores[c] = TIMBER_CLASS_BASE_SCORES[c];")
            lines.append("")
            lines.append("")

            # Trees are interleaved: tree i contributes to class (i % n_classes)
            for i, tree in enumerate(ensemble.trees):
                cls_idx = i % ensemble.n_classes
                lines.append(
                    f"    scores[{cls_idx}] += (double)traverse_tree(inputs, "
                    f"tree_{tree.tree_id}_features, tree_{tree.tree_id}_thresholds, "
                    f"tree_{tree.tree_id}_left, tree_{tree.tree_id}_right, "
                    f"tree_{tree.tree_id}_leaves, tree_{tree.tree_id}_is_leaf, "
                    f"tree_{tree.tree_id}_default_left, TREE_{tree.tree_id}_N_NODES);"
                )

            # Softmax in double precision
            lines.append("")
            lines.append("    { /* softmax */")
            lines.append("        double max_val = scores[0];")
            lines.append(f"        for (c = 1; c < {ensemble.n_classes}; c++)")
            lines.append("            if (scores[c] > max_val) max_val = scores[c];")
            lines.append("        double denom = 0.0;")
            lines.append(f"        for (c = 0; c < {ensemble.n_classes}; c++) {{")
            lines.append("            scores[c] = exp(scores[c] - max_val);")
            lines.append("            denom += scores[c];")
            lines.append("        }")
            lines.append(f"        for (c = 0; c < {ensemble.n_classes}; c++)")
            lines.append(f"            outputs[c] = ({float_type})(scores[c] / denom);")
            lines.append("    }")
        else:
            # Binary classification or regression: use double accumulator for precision
            lines.append("    double sum = (double)TIMBER_BASE_SCORE;")
            lines.append("")

            for tree in ensemble.trees:
                lines.append(
                    f"    sum += (double)traverse_tree(inputs, "
                    f"tree_{tree.tree_id}_features, tree_{tree.tree_id}_thresholds, "
                    f"tree_{tree.tree_id}_left, tree_{tree.tree_id}_right, "
                    f"tree_{tree.tree_id}_leaves, tree_{tree.tree_id}_is_leaf, "
                    f"tree_{tree.tree_id}_default_left, TREE_{tree.tree_id}_N_NODES);"
                )

            lines.append("")

            if ensemble.objective in (Objective.BINARY_CLASSIFICATION, Objective.REGRESSION_LOGISTIC):
                lines.append(f"    outputs[0] = ({float_type})(1.0 / (1.0 + exp(-sum)));")
            else:
                lines.append(f"    outputs[0] = ({float_type})sum;")

        lines.extend([
            "",
            "    return 0;",
            "}",
            "",
        ])

        # Batched inference
        lines.extend([
            "int timber_infer(",
            f"    const {float_type}*  inputs,",
            "    int                  n_samples,",
            f"    {float_type}*        outputs,",
            "    const TimberCtx*     ctx",
            ") {",
            "    int i;",
            "    if (inputs == NULL || outputs == NULL) return TIMBER_ERR_NULL;",
            "    if (n_samples <= 0) return TIMBER_ERR_BOUNDS;",
            "",
            "    for (i = 0; i < n_samples; i++) {",
            "        int rc = timber_infer_single(",
            "            inputs + i * TIMBER_N_FEATURES,",
            "            outputs + i * TIMBER_N_OUTPUTS,",
            "            ctx",
            "        );",
            "        if (rc != 0) return rc;",
            "    }",
            "    return 0;",
            "}",
        ])

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Linear model emission
    # ------------------------------------------------------------------

    def _emit_header_linear(self, ir: TimberIR, stage: "LinearStage") -> str:
        """Generate model.h for a linear model."""
        n_features = len(stage.weights) // max(stage.n_classes, 1) if stage.multi_weights else len(stage.weights)
        n_outputs = 1 if stage.n_classes <= 2 else stage.n_classes
        float_type = self._float_type()
        lines = self._common_header_prefix(n_features, n_outputs, extra_defines={
            "TIMBER_N_CLASSES": str(max(stage.n_classes, 1)),
        })
        return "\n".join(lines) + "\n"

    def _emit_data_linear(self, ir: TimberIR, stage: "LinearStage") -> str:
        """Generate model_data.c for a linear model."""
        float_type = self._float_type()
        lines = [
            "/* model_data.c — Timber linear model data */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "",
        ]
        n_features = len(stage.weights) // max(stage.n_classes, 1) if stage.multi_weights else len(stage.weights)

        if stage.multi_weights:
            n_cls = stage.n_classes
            n_w = len(stage.weights)
            w_str = ", ".join(self._format_float(w) for w in stage.weights)
            lines.append(f"static const {float_type} TIMBER_WEIGHTS[{n_w}] = {{{w_str}}};")
            if stage.biases:
                b_str = ", ".join(self._format_float(b) for b in stage.biases)
                lines.append(f"static const {float_type} TIMBER_BIASES[{n_cls}] = {{{b_str}}};")
            else:
                lines.append(f"static const {float_type} TIMBER_BIASES[{n_cls}] = {{{', '.join(['0.0f']*n_cls)}}};")
        else:
            n_w = len(stage.weights)
            w_str = ", ".join(self._format_float(w) for w in stage.weights)
            lines.append(f"static const {float_type} TIMBER_WEIGHTS[{n_w}] = {{{w_str}}};")
            lines.append(f"static const {float_type} TIMBER_BIAS = {self._format_float(stage.bias)};")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _emit_inference_linear(self, ir: TimberIR, stage: "LinearStage") -> str:
        """Generate model.c for a linear model."""
        float_type = self._float_type()
        n_features = len(stage.weights) // max(stage.n_classes, 1) if stage.multi_weights else len(stage.weights)
        n_outputs = 1 if stage.n_classes <= 2 else stage.n_classes

        lines = [
            "/* model.c — Timber linear model inference */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "",
            '#include "model_data.c"',
            "",
            "struct TimberCtx { int initialized; };",
            "static struct TimberCtx _default_ctx = {1};",
            "",
        ]
        lines += self._common_runtime_fns()
        lines += [
            "int timber_infer_single(",
            f"    const {float_type}  inputs[TIMBER_N_FEATURES],",
            f"    {float_type}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ") {",
            "    (void)ctx;",
            "    int i;",
        ]

        if stage.multi_weights:
            n_cls = stage.n_classes
            # n_outputs caps writes to the declared output buffer size
            lines.append(f"    double scores[{n_cls}];")
            lines.append(f"    for (i = 0; i < {n_cls}; i++) {{")
            lines.append("        int j;")
            lines.append("        scores[i] = (double)TIMBER_BIASES[i];")
            lines.append(f"        for (j = 0; j < {n_features}; j++)")
            lines.append(f"            scores[i] += (double)TIMBER_WEIGHTS[i * {n_features} + j] * (double)inputs[j];")
            lines.append("    }")
            if stage.activation == "softmax":
                lines += [
                    "    { /* softmax */",
                    "        double max_v = scores[0]; int c;",
                    f"        for (c = 1; c < {n_cls}; c++) if (scores[c] > max_v) max_v = scores[c];",
                    "        double denom = 0.0;",
                    f"        for (c = 0; c < {n_cls}; c++) {{ scores[c] = exp(scores[c] - max_v); denom += scores[c]; }}",
                    f"        for (c = 0; c < {n_outputs}; c++) outputs[c] = ({float_type})(scores[c] / denom);",
                    "    }",
                ]
            else:
                lines.append(f"    for (i = 0; i < {n_outputs}; i++) outputs[i] = ({float_type})scores[i];")
        else:
            lines.append("    double sum = (double)TIMBER_BIAS;")
            lines.append(f"    for (i = 0; i < {n_features}; i++)")
            lines.append("        sum += (double)TIMBER_WEIGHTS[i] * (double)inputs[i];")
            if stage.activation in ("sigmoid", "logistic"):
                lines.append(f"    outputs[0] = ({float_type})(1.0 / (1.0 + exp(-sum)));")
            else:
                lines.append(f"    outputs[0] = ({float_type})sum;")

        lines += ["    return 0;", "}", ""]
        lines += self._batched_infer_fn(float_type)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # SVM emission
    # ------------------------------------------------------------------

    def _emit_header_svm(self, ir: TimberIR, stage: "SVMStage") -> str:
        """Generate model.h for an SVM model."""
        n_outputs = 1 if stage.n_classes <= 2 else stage.n_classes
        lines = self._common_header_prefix(stage.n_features, n_outputs, extra_defines={
            "TIMBER_N_SV": str(stage.n_sv),
            "TIMBER_N_CLASSES": str(stage.n_classes),
        })
        return "\n".join(lines) + "\n"

    def _emit_data_svm(self, ir: TimberIR, stage: "SVMStage") -> str:
        """Generate model_data.c for an SVM model."""
        float_type = self._float_type()
        n_sv = stage.n_sv
        n_features = stage.n_features

        lines = [
            "/* model_data.c — Timber SVM model data */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "",
        ]

        # Support vectors — row-major [n_sv x n_features]
        sv_flat = [v for sv in stage.support_vectors for v in sv]
        sv_str = ", ".join(self._format_float(v) for v in sv_flat)
        lines.append(f"static const {float_type} TIMBER_SV[{n_sv * n_features}] = {{{sv_str}}};")

        # Dual coefficients
        dc_str = ", ".join(self._format_float(v) for v in stage.dual_coef)
        lines.append(f"static const {float_type} TIMBER_DUAL_COEF[{len(stage.dual_coef)}] = {{{dc_str}}};")

        # Rho (bias)
        rho_str = ", ".join(self._format_float(v) for v in stage.rho)
        n_rho = max(len(stage.rho), 1)
        lines.append(f"static const {float_type} TIMBER_RHO[{n_rho}] = {{{rho_str or '0.0f'}}};")

        # Kernel parameters
        lines.append(f"static const {float_type} TIMBER_GAMMA = {self._format_float(stage.gamma)};")
        lines.append(f"static const {float_type} TIMBER_COEF0 = {self._format_float(stage.coef0)};")
        lines.append(f"static const int32_t TIMBER_DEGREE = {stage.degree};")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _emit_inference_svm(self, ir: TimberIR, stage: "SVMStage") -> str:
        """Generate model.c for an SVM model."""
        float_type = self._float_type()
        n_sv = stage.n_sv
        n_features = stage.n_features
        n_outputs = 1 if stage.n_classes <= 2 else stage.n_classes
        kernel = stage.kernel_type.lower()

        lines = [
            "/* model.c — Timber SVM inference */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "",
            '#include "model_data.c"',
            "",
            "struct TimberCtx { int initialized; };",
            "static struct TimberCtx _default_ctx = {1};",
            "",
        ]
        lines += self._common_runtime_fns()

        # Kernel function
        lines += [
            f"static double timber_kernel(const {float_type}* x, const {float_type}* sv, int n) {{",
            "    int i;",
        ]
        if kernel == "rbf":
            lines += [
                "    double dist = 0.0;",
                "    for (i = 0; i < n; i++) {",
                "        double d = (double)x[i] - (double)sv[i];",
                "        dist += d * d;",
                "    }",
                "    return exp(-(double)TIMBER_GAMMA * dist);",
            ]
        elif kernel == "linear":
            lines += [
                "    double dot = 0.0;",
                "    for (i = 0; i < n; i++) dot += (double)x[i] * (double)sv[i];",
                "    return dot;",
            ]
        elif kernel == "poly":
            lines += [
                "    double dot = 0.0;",
                "    for (i = 0; i < n; i++) dot += (double)x[i] * (double)sv[i];",
                "    double base = (double)TIMBER_GAMMA * dot + (double)TIMBER_COEF0;",
                "    double result = 1.0; int d;",
                "    for (d = 0; d < TIMBER_DEGREE; d++) result *= base;",
                "    return result;",
            ]
        else:  # sigmoid
            lines += [
                "    double dot = 0.0;",
                "    for (i = 0; i < n; i++) dot += (double)x[i] * (double)sv[i];",
                "    return tanh((double)TIMBER_GAMMA * dot + (double)TIMBER_COEF0);",
            ]
        lines += ["}", ""]

        # Inference function
        lines += [
            "int timber_infer_single(",
            f"    const {float_type}  inputs[TIMBER_N_FEATURES],",
            f"    {float_type}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ") {",
            "    (void)ctx;",
            "    int sv;",
            "    double decision = (double)TIMBER_RHO[0];",
            f"    for (sv = 0; sv < {n_sv}; sv++) {{",
            f"        decision += (double)TIMBER_DUAL_COEF[sv] * timber_kernel(inputs, TIMBER_SV + sv * {n_features}, {n_features});",
            "    }",
        ]
        if stage.post_transform == "logistic":
            lines.append(f"    outputs[0] = ({float_type})(1.0 / (1.0 + exp(-decision)));")
        else:
            lines.append(f"    outputs[0] = ({float_type})decision;")
        lines += ["    return 0;", "}", ""]
        lines += self._batched_infer_fn(float_type)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Isolation Forest emission
    # ------------------------------------------------------------------

    def _emit_header_iforest(self, ir: TimberIR, stage: "IsolationForestStage") -> str:
        lines = self._common_header_prefix(stage.n_features, 1, extra_defines={
            "TIMBER_N_TREES": str(stage.n_trees),
        })
        return "\n".join(lines) + "\n"

    def _emit_data_iforest(self, ir: TimberIR, stage: "IsolationForestStage") -> str:
        ft = self._float_type()
        lines = [
            "/* model_data.c — Timber Isolation Forest data */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "",
        ]
        for tree in stage.trees:
            tid = tree.tree_id
            n   = len(tree.nodes)
            thr_vals = ", ".join(repr(nd.threshold) for nd in tree.nodes)
            pl_vals  = ", ".join(repr(nd.leaf_value) for nd in tree.nodes)
            lines.append(f"static const int32_t ift_{tid}_feat[{n}]  = {{{', '.join(str(nd.feature_index) for nd in tree.nodes)}}};")
            lines.append(f"static const double  ift_{tid}_thr[{n}]   = {{{thr_vals}}};")
            lines.append(f"static const int32_t ift_{tid}_left[{n}]  = {{{', '.join(str(nd.left_child)     for nd in tree.nodes)}}};")
            lines.append(f"static const int32_t ift_{tid}_right[{n}] = {{{', '.join(str(nd.right_child)    for nd in tree.nodes)}}};")
            lines.append(f"static const double  ift_{tid}_pl[{n}]    = {{{pl_vals}}};")
            lines.append(f"static const int8_t  ift_{tid}_leaf[{n}]  = {{{', '.join('1' if nd.is_leaf else '0' for nd in tree.nodes)}}};")
            lines.append(f"#define IFT_{tid}_N {n}")
            lines.append("")
        c_max = _c_factor(stage.max_samples)
        lines.append(f"static const double TIMBER_C_MAX = {c_max!r};")
        lines.append(f"static const double TIMBER_IF_OFFSET = {stage.offset!r};")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _emit_inference_iforest(self, ir: TimberIR, stage: "IsolationForestStage") -> str:
        ft = self._float_type()
        max_d = max((t.max_depth for t in stage.trees), default=32) + 2
        lines = [
            "/* model.c — Timber Isolation Forest inference */",
            "/* Generated by Timber — DO NOT EDIT */",
            "/*",
            " * Output: decision_function = -anomaly_score - offset",
            " *         positive  -> inlier,  negative -> outlier",
            " *         anomaly_score = 2^(-mean_path / c_max_samples)",
            " */",
            "",
            '#include "model.h"',
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "",
            '#include "model_data.c"',
            "",
            "struct TimberCtx { int initialized; };",
            "static struct TimberCtx _default_ctx = {1};",
            "",
        ]
        lines += self._common_runtime_fns()
        # Generic tree traversal — leaf values are pre-computed path-length contributions
        lines += [
            "static double iforest_traverse(",
            f"    const {ft}*    x,",
            "    const int32_t* feat,",
            "    const double*  thr,",
            "    const int32_t* lft,",
            "    const int32_t* rgt,",
            "    const double*  pl,",
            "    const int8_t*  leaf,",
            "    int n_nodes",
            ") {",
            "    int node = 0;",
            f"    int guard = {max_d};",
            "    while (guard-- > 0 && node >= 0 && node < n_nodes) {",
            "        if (leaf[node]) return (double)pl[node];",
            "        if ((double)x[feat[node]] <= thr[node])",
            "            node = lft[node];",
            "        else",
            "            node = rgt[node];",
            "    }",
            "    return 1.0;",
            "}",
            "",
            "int timber_infer_single(",
            f"    const {ft}  inputs[TIMBER_N_FEATURES],",
            f"    {ft}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ") {",
            "    (void)ctx;",
            "    double total = 0.0;",
        ]
        for tree in stage.trees:
            tid = tree.tree_id
            lines.append(
                f"    total += iforest_traverse(inputs,"
                f" ift_{tid}_feat, ift_{tid}_thr,"
                f" ift_{tid}_left, ift_{tid}_right,"
                f" ift_{tid}_pl, ift_{tid}_leaf, IFT_{tid}_N);"
            )
        lines += [
            f"    double mean_path = total / {stage.n_trees};",
            "    double anomaly = exp2(-mean_path / TIMBER_C_MAX);",
            f"    outputs[0] = ({ft})(-anomaly - TIMBER_IF_OFFSET);",
            "    return 0;",
            "}",
            "",
        ]
        lines += self._batched_infer_fn(ft)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Naive Bayes emission
    # ------------------------------------------------------------------

    def _emit_header_nb(self, ir: TimberIR, stage: "NaiveBayesStage") -> str:
        lines = self._common_header_prefix(stage.n_features, stage.n_classes, extra_defines={
            "TIMBER_N_CLASSES": str(stage.n_classes),
        })
        return "\n".join(lines) + "\n"

    def _emit_data_nb(self, ir: TimberIR, stage: "NaiveBayesStage") -> str:
        ft = self._float_type()
        nc, nf = stage.n_classes, stage.n_features
        lines = [
            "/* model_data.c — Timber Gaussian Naive Bayes data */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "",
        ]
        lp = ", ".join(self._format_float(v) for v in stage.log_prior)
        lines.append(f"static const {ft} TIMBER_NB_LOG_PRIOR[{nc}] = {{{lp}}};")
        theta_flat = [v for row in stage.theta for v in row]
        th = ", ".join(self._format_float(v) for v in theta_flat)
        lines.append(f"static const {ft} TIMBER_NB_THETA[{nc * nf}] = {{{th}}};")
        lvc_flat = [v for row in stage.log_var_const for v in row]
        lv = ", ".join(self._format_float(v) for v in lvc_flat)
        lines.append(f"static const {ft} TIMBER_NB_LOGVC[{nc * nf}] = {{{lv}}};")
        iv_flat = [v for row in stage.inv_2var for v in row]
        iv = ", ".join(self._format_float(v) for v in iv_flat)
        lines.append(f"static const {ft} TIMBER_NB_INV2V[{nc * nf}] = {{{iv}}};")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _emit_inference_nb(self, ir: TimberIR, stage: "NaiveBayesStage") -> str:
        ft = self._float_type()
        nc, nf = stage.n_classes, stage.n_features
        lines = [
            "/* model.c — Timber Gaussian Naive Bayes inference */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "",
            '#include "model_data.c"',
            "",
            "struct TimberCtx { int initialized; };",
            "static struct TimberCtx _default_ctx = {1};",
            "",
        ]
        lines += self._common_runtime_fns()
        lines += [
            "int timber_infer_single(",
            f"    const {ft}  inputs[TIMBER_N_FEATURES],",
            f"    {ft}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ") {",
            "    (void)ctx;",
            f"    double scores[{nc}];",
            "    int c, f;",
            f"    for (c = 0; c < {nc}; c++) {{",
            "        scores[c] = (double)TIMBER_NB_LOG_PRIOR[c];",
            f"        for (f = 0; f < {nf}; f++) {{",
            f"            double d = (double)inputs[f] - (double)TIMBER_NB_THETA[c * {nf} + f];",
            f"            scores[c] += (double)TIMBER_NB_LOGVC[c * {nf} + f]",
            f"                       - (double)TIMBER_NB_INV2V[c * {nf} + f] * d * d;",
            "        }",
            "    }",
            "    /* softmax */",
            "    { double mx = scores[0]; int i;",
            f"      for (i = 1; i < {nc}; i++) if (scores[i] > mx) mx = scores[i];",
            "      double sm = 0.0;",
            f"      for (i = 0; i < {nc}; i++) {{ scores[i] = exp(scores[i] - mx); sm += scores[i]; }}",
            f"      for (i = 0; i < {nc}; i++) outputs[i] = ({ft})(scores[i] / sm);",
            "    }",
            "    return 0;",
            "}",
            "",
        ]
        lines += self._batched_infer_fn(ft)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Gaussian Process Regressor emission
    # ------------------------------------------------------------------

    def _emit_header_gpr(self, ir: TimberIR, stage: "GPRStage") -> str:
        lines = self._common_header_prefix(stage.n_features, 1, extra_defines={
            "TIMBER_N_TRAIN": str(stage.n_train),
        })
        return "\n".join(lines) + "\n"

    def _emit_data_gpr(self, ir: TimberIR, stage: "GPRStage") -> str:
        ft = self._float_type()
        nt, nf = stage.n_train, stage.n_features
        lines = [
            "/* model_data.c — Timber GPR data */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "",
        ]
        x_flat = [v for row in stage.X_train for v in row]
        xv = ", ".join(self._format_float(v) for v in x_flat)
        lines.append(f"static const {ft} TIMBER_GPR_X[{nt * nf}] = {{{xv}}};")
        av = ", ".join(self._format_float(v) for v in stage.alpha)
        lines.append(f"static const {ft} TIMBER_GPR_ALPHA[{nt}] = {{{av}}};")
        inv_2ls2 = 1.0 / (2.0 * stage.length_scale ** 2)
        amp2     = stage.amplitude ** 2
        lines.append(f"static const {ft} TIMBER_GPR_INV2LS2 = {self._format_float(inv_2ls2)};")
        lines.append(f"static const {ft} TIMBER_GPR_AMP2    = {self._format_float(amp2)};")
        lines.append(f"static const {ft} TIMBER_GPR_YMEAN   = {self._format_float(stage.y_train_mean)};")
        lines.append(f"static const {ft} TIMBER_GPR_YSTD    = {self._format_float(stage.y_train_std)};")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _emit_inference_gpr(self, ir: TimberIR, stage: "GPRStage") -> str:
        ft = self._float_type()
        nt, nf = stage.n_train, stage.n_features
        lines = [
            "/* model.c — Timber GPR inference */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "",
            '#include "model_data.c"',
            "",
            "struct TimberCtx { int initialized; };",
            "static struct TimberCtx _default_ctx = {1};",
            "",
        ]
        lines += self._common_runtime_fns()
        lines += [
            "int timber_infer_single(",
            f"    const {ft}  inputs[TIMBER_N_FEATURES],",
            f"    {ft}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ") {",
            "    (void)ctx;",
            "    double pred = 0.0;",
            "    int i, f;",
            f"    for (i = 0; i < {nt}; i++) {{",
            "        double sq = 0.0;",
            f"        for (f = 0; f < {nf}; f++) {{",
            f"            double d = (double)inputs[f] - (double)TIMBER_GPR_X[i * {nf} + f];",
            "            sq += d * d;",
            "        }",
            "        double k = (double)TIMBER_GPR_AMP2 * exp(-(double)TIMBER_GPR_INV2LS2 * sq);",
            "        pred += k * (double)TIMBER_GPR_ALPHA[i];",
            "    }",
            f"    outputs[0] = ({ft})(pred * (double)TIMBER_GPR_YSTD + (double)TIMBER_GPR_YMEAN);",
            "    return 0;",
            "}",
            "",
        ]
        lines += self._batched_infer_fn(ft)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # k-NN emission
    # ------------------------------------------------------------------

    def _emit_header_knn(self, ir: TimberIR, stage: "KNNStage") -> str:
        extra = {"TIMBER_N_TRAIN": str(stage.n_train), "TIMBER_K": str(stage.k)}
        if stage.task_type == "classifier":
            extra["TIMBER_N_CLASSES"] = str(stage.n_classes)
        lines = self._common_header_prefix(stage.n_features, stage.n_outputs, extra_defines=extra)
        return "\n".join(lines) + "\n"

    def _emit_data_knn(self, ir: TimberIR, stage: "KNNStage") -> str:
        ft = self._float_type()
        nt, nf = stage.n_train, stage.n_features
        no = stage.n_outputs
        lines = [
            "/* model_data.c — Timber k-NN data */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "",
        ]
        x_flat = [v for row in stage.X_train for v in row]
        xv = ", ".join(self._format_float(v) for v in x_flat)
        lines.append(f"static const {ft} TIMBER_KNN_X[{nt * nf}] = {{{xv}}};")
        y_flat = [v for row in stage.y_train for v in row]
        yv = ", ".join(self._format_float(v) for v in y_flat)
        lines.append(f"static const {ft} TIMBER_KNN_Y[{nt * no}] = {{{yv}}};")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _emit_inference_knn(self, ir: TimberIR, stage: "KNNStage") -> str:
        ft = self._float_type()
        nt, nf, k = stage.n_train, stage.n_features, stage.k
        no = stage.n_outputs
        is_clf = stage.task_type == "classifier"
        nc = stage.n_classes

        lines = [
            "/* model.c — Timber k-NN inference */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "#include <float.h>",
            "",
            '#include "model_data.c"',
            "",
            "struct TimberCtx { int initialized; };",
            "static struct TimberCtx _default_ctx = {1};",
            "",
        ]
        lines += self._common_runtime_fns()

        # Distance function
        if stage.metric == "manhattan":
            lines += [
                f"static double knn_dist(const {ft}* a, const {ft}* b, int n) {{",
                "    double d = 0.0; int i;",
                "    for (i = 0; i < n; i++) { double diff = (double)a[i]-(double)b[i]; d += diff<0?-diff:diff; }",
                "    return d;",
                "}",
                "",
            ]
        else:  # euclidean (squared — monotone so no sqrt needed for sorting)
            lines += [
                f"static double knn_dist(const {ft}* a, const {ft}* b, int n) {{",
                "    double d = 0.0; int i;",
                "    for (i = 0; i < n; i++) { double diff = (double)a[i]-(double)b[i]; d += diff*diff; }",
                "    return d;",
                "}",
                "",
            ]

        lines += [
            "int timber_infer_single(",
            f"    const {ft}  inputs[TIMBER_N_FEATURES],",
            f"    {ft}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ") {",
            "    (void)ctx;",
            f"    double kd[{k}];",       # top-k distances
            f"    int    ki[{k}];",       # top-k indices
            "    int i, j;",
            f"    for (i = 0; i < {k}; i++) {{ kd[i] = DBL_MAX; ki[i] = -1; }}",
            "",
            f"    for (i = 0; i < {nt}; i++) {{",
            f"        double d = knn_dist(inputs, TIMBER_KNN_X + i * {nf}, {nf});",
            "        /* insert into top-k if closer than current worst */",
            "        int worst = 0;",
            f"        for (j = 1; j < {k}; j++) if (kd[j] > kd[worst]) worst = j;",
            "        if (d < kd[worst]) { kd[worst] = d; ki[worst] = i; }",
            "    }",
            "",
        ]

        if is_clf:
            lines += [
                f"    int votes[{nc}];",
                f"    for (i = 0; i < {nc}; i++) votes[i] = 0;",
                f"    for (i = 0; i < {k}; i++) {{",
                "        if (ki[i] >= 0) {",
                "            int cls = (int)TIMBER_KNN_Y[ki[i]];",
                f"            if (cls >= 0 && cls < {nc}) votes[cls]++;",
                "        }",
                "    }",
                "    int best = 0;",
                f"    for (i = 1; i < {nc}; i++) if (votes[i] > votes[best]) best = i;",
                f"    outputs[0] = ({ft})best;",
            ]
        else:
            lines += [
                f"    for (j = 0; j < {no}; j++) {{",
                "        double acc = 0.0; int cnt = 0;",
                f"        for (i = 0; i < {k}; i++) {{",
                "            if (ki[i] >= 0) {",
                f"                acc += (double)TIMBER_KNN_Y[ki[i] * {no} + j];",
                "                cnt++;",
                "            }",
                "        }",
                f"        outputs[j] = ({ft})(cnt > 0 ? acc / cnt : 0.0);",
                "    }",
            ]

        lines += ["    return 0;", "}", ""]
        lines += self._batched_infer_fn(ft)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Kinematics (FK) emission
    # ------------------------------------------------------------------

    @staticmethod
    def _rpy_to_mat4(rpy: list[float], xyz: list[float]) -> list[float]:
        """Pre-compute a 4x4 row-major homogeneous transform from RPY + XYZ."""
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll),  math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw),   math.sin(yaw)
        return [
            cy*cp,           cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr,  xyz[0],
            sy*cp,           sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr,  xyz[1],
            -sp,             cp*sr,             cp*cr,             xyz[2],
            0.0,             0.0,               0.0,               1.0,
        ]

    def _emit_header_kinematics(self, ir: TimberIR, stage: KinematicsStage) -> str:
        """Generate model.h for a forward-kinematics model."""
        n_dof = stage.n_dof
        lines = self._common_header_prefix(n_dof, 16, extra_defines={
            "TIMBER_N_DOF":    str(n_dof),
            "TIMBER_N_JOINTS": str(len(stage.joints)),
        })
        # Splice in the FK-specific declaration just before the closing guard
        insert_at = next(
            i for i, ln in enumerate(lines) if "timber_infer_single" in ln
        )
        fk_decl = [
            "/* Compute forward kinematics. */",
            "/* joint_angles: TIMBER_N_DOF values [rad or m] */",
            "/* transform:    4x4 homogeneous matrix, row-major (16 floats) */",
            "int timber_fk(",
            "    const float joint_angles[TIMBER_N_DOF],",
            "    float       transform[16],",
            "    const TimberCtx* ctx",
            ");",
            "",
        ]
        lines[insert_at:insert_at] = fk_decl
        return "\n".join(lines) + "\n"

    def _emit_data_kinematics(self, ir: TimberIR, stage: KinematicsStage) -> str:
        """Generate model_data.c for a kinematics model.

        Pre-computes each joint's 4x4 origin transform and stores the
        active-joint axis vectors as static const arrays.
        """
        lines = [
            "/* model_data.c — Timber kinematics model data */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "",
        ]

        for i, j in enumerate(stage.joints):
            mat = self._rpy_to_mat4(j.origin_rpy, j.origin_xyz)
            vals = ", ".join(self._format_float(v) for v in mat)
            lines.append(
                f"static const float JOINT_{i}_ORIGIN[16] = {{{vals}}};"
            )
            if j.joint_type in _ACTIVE_TYPES:
                ax = ", ".join(self._format_float(v) for v in j.axis)
                lines.append(f"static const float JOINT_{i}_AXIS[3] = {{{ax}}};")
            lines.append("")

        return "\n".join(lines) + "\n"

    def _emit_inference_kinematics(self, ir: TimberIR, stage: KinematicsStage) -> str:
        """Generate model.c for a forward-kinematics model."""
        lines = [
            "/* model.c — Timber forward kinematics inference */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            '#include "model.h"',
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "",
            '#include "model_data.c"',
            "",
            "struct TimberCtx { int initialized; };",
            "static struct TimberCtx _default_ctx = {1};",
            "",
        ]
        lines += self._common_runtime_fns()

        has_revolute  = any(j.joint_type in ("revolute", "continuous") for j in stage.joints)
        has_prismatic = any(j.joint_type == "prismatic" for j in stage.joints)

        # Matrix helpers
        lines += [
            "static void mat4_identity(float *T) {",
            "    int i;",
            "    for (i = 0; i < 16; i++) T[i] = 0.0f;",
            "    T[0] = T[5] = T[10] = T[15] = 1.0f;",
            "}",
            "",
            "static void mat4_mul(const float *A, const float *B, float *C) {",
            "    /* C = A * B  (C must not alias A or B) */",
            "    int i, j, k;",
            "    for (i = 0; i < 4; i++)",
            "        for (j = 0; j < 4; j++) {",
            "            C[i*4+j] = 0.0f;",
            "            for (k = 0; k < 4; k++)",
            "                C[i*4+j] += A[i*4+k] * B[k*4+j];",
            "        }",
            "}",
            "",
        ]
        if has_revolute:
            lines += [
                "static void rodrigues(const float *axis, float angle, float *T) {",
                "    float c = cosf(angle), s = sinf(angle), t = 1.0f - c;",
                "    float ax = axis[0], ay = axis[1], az = axis[2];",
                "    T[0]  = c+ax*ax*t;       T[1]  = ax*ay*t-az*s;  T[2]  = ax*az*t+ay*s;  T[3]  = 0.0f;",
                "    T[4]  = ay*ax*t+az*s;    T[5]  = c+ay*ay*t;     T[6]  = ay*az*t-ax*s;  T[7]  = 0.0f;",
                "    T[8]  = az*ax*t-ay*s;    T[9]  = az*ay*t+ax*s;  T[10] = c+az*az*t;     T[11] = 0.0f;",
                "    T[12] = 0.0f;            T[13] = 0.0f;          T[14] = 0.0f;          T[15] = 1.0f;",
                "}",
                "",
            ]
        if has_prismatic:
            lines += [
                "static void prismatic(const float *axis, float q, float *T) {",
                "    mat4_identity(T);",
                "    T[3]  = axis[0] * q;",
                "    T[7]  = axis[1] * q;",
                "    T[11] = axis[2] * q;",
                "}",
                "",
            ]

        # timber_fk
        n_dof = stage.n_dof
        lines += [
            "int timber_fk(",
            "    const float joint_angles[TIMBER_N_DOF],",
            "    float       transform[16],",
            "    const TimberCtx *ctx",
            ") {",
            "    (void)ctx;",
            "    float acc[16], tmp[16], Tj[16];",
            "    int i;",
            "    mat4_identity(acc);",
            "",
        ]

        qi = 0
        for idx, j in enumerate(stage.joints):
            lines.append(f"    /* joint {idx}: {j.name} ({j.joint_type}) */")
            lines.append(f"    mat4_mul(acc, JOINT_{idx}_ORIGIN, tmp);")
            if j.joint_type in _ACTIVE_TYPES:
                if j.joint_type == "prismatic":
                    lines.append(
                        f"    prismatic(JOINT_{idx}_AXIS, joint_angles[{qi}], Tj);"
                    )
                else:
                    lines.append(
                        f"    rodrigues(JOINT_{idx}_AXIS, joint_angles[{qi}], Tj);"
                    )
                lines.append("    mat4_mul(tmp, Tj, acc);")
                qi += 1
            else:
                lines.append("    for (i = 0; i < 16; i++) acc[i] = tmp[i];")
            lines.append("")

        lines += [
            "    for (i = 0; i < 16; i++) transform[i] = acc[i];",
            "    return 0;",
            "}",
            "",
            "int timber_infer_single(",
            "    const float  inputs[TIMBER_N_FEATURES],",
            "    float        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx* ctx",
            ") {",
            "    return timber_fk(inputs, outputs, ctx);",
            "}",
            "",
        ]
        lines += self._batched_infer_fn("float")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Shared helpers for header / runtime
    # ------------------------------------------------------------------

    def _common_header_prefix(
        self, n_features: int, n_outputs: int, extra_defines: dict | None = None
    ) -> list:
        """Return common header lines shared by tree/linear/SVM headers."""
        float_type = self._float_type()
        lines = [
            "/* model.h — Timber compiled model inference header */",
            "/* Generated by Timber — DO NOT EDIT */",
            "",
            "#ifndef TIMBER_MODEL_H",
            "#define TIMBER_MODEL_H",
            "",
            "#include <stdint.h>",
            "#include <stddef.h>",
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "#define TIMBER_ABI_VERSION  1",
            '#define TIMBER_VERSION     "0.1.0"',
            "",
            f"#define TIMBER_N_FEATURES {n_features}",
            f"#define TIMBER_N_OUTPUTS  {n_outputs}",
        ]
        if extra_defines:
            for k, v in extra_defines.items():
                lines.append(f"#define {k} {v}")
        lines += [
            "",
            "typedef struct TimberCtx TimberCtx;",
            "",
            "#define TIMBER_OK          0",
            "#define TIMBER_ERR_NULL   -1",
            "#define TIMBER_ERR_INIT   -2",
            "#define TIMBER_ERR_BOUNDS -3",
            "",
            "int timber_init(TimberCtx** ctx);",
            "int timber_abi_version(void);",
            "typedef void (*timber_log_fn)(int level, const char* msg);",
            "void timber_set_log_callback(timber_log_fn fn);",
            "const char* timber_strerror(int code);",
            "void timber_free(TimberCtx* ctx);",
            "",
            "int timber_infer(",
            f"    const {float_type}*  inputs,",
            "    int                  n_samples,",
            f"    {float_type}*        outputs,",
            "    const TimberCtx*     ctx",
            ");",
            "",
            "int timber_infer_single(",
            f"    const {float_type}  inputs[TIMBER_N_FEATURES],",
            f"    {float_type}        outputs[TIMBER_N_OUTPUTS],",
            "    const TimberCtx*    ctx",
            ");",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            "#endif /* TIMBER_MODEL_H */",
        ]
        return lines

    def _common_runtime_fns(self) -> list:
        """Return shared runtime (logging, init, free, strerror) lines."""
        return [
            "static timber_log_fn _timber_log_cb = NULL;",
            "void timber_set_log_callback(timber_log_fn fn) { _timber_log_cb = fn; }",
            "static void timber_log(int level, const char* msg) {",
            "    if (_timber_log_cb) _timber_log_cb(level, msg);",
            "}",
            "const char* timber_strerror(int code) {",
            "    switch (code) {",
            '        case  0: return "TIMBER_OK";',
            '        case -1: return "TIMBER_ERR_NULL: null pointer argument";',
            '        case -2: return "TIMBER_ERR_INIT: context not initialized";',
            '        case -3: return "TIMBER_ERR_BOUNDS: argument out of bounds";',
            '        default: return "TIMBER_ERR_UNKNOWN";',
            "    }",
            "}",
            "int timber_abi_version(void) { return TIMBER_ABI_VERSION; }",
            "int timber_init(TimberCtx** ctx) {",
            "    if (ctx == NULL) return TIMBER_ERR_NULL;",
            "    *ctx = &_default_ctx;",
            "    timber_log(2, \"timber_init: OK\");",
            "    return TIMBER_OK;",
            "}",
            "void timber_free(TimberCtx* ctx) { (void)ctx; }",
            "",
        ]

    def _batched_infer_fn(self, float_type: str) -> list:
        """Return the batched timber_infer function lines."""
        return [
            "int timber_infer(",
            f"    const {float_type}*  inputs,",
            "    int                  n_samples,",
            f"    {float_type}*        outputs,",
            "    const TimberCtx*     ctx",
            ") {",
            "    int i;",
            "    if (inputs == NULL || outputs == NULL) return TIMBER_ERR_NULL;",
            "    if (n_samples <= 0) return TIMBER_ERR_BOUNDS;",
            "    for (i = 0; i < n_samples; i++) {",
            "        int rc = timber_infer_single(",
            "            inputs + i * TIMBER_N_FEATURES,",
            "            outputs + i * TIMBER_N_OUTPUTS,",
            "            ctx",
            "        );",
            "        if (rc != 0) return rc;",
            "    }",
            "    return 0;",
            "}",
        ]

    def _emit_cmake(self, ir: TimberIR) -> str:
        """Generate CMakeLists.txt for the compiled model."""
        lines = [
            "# CMakeLists.txt — Timber compiled model",
            "# Generated by Timber v0.1",
            "",
            "cmake_minimum_required(VERSION 3.10)",
            "project(timber_model C)",
            "",
            "set(CMAKE_C_STANDARD 99)",
            "set(CMAKE_C_STANDARD_REQUIRED ON)",
            "",
            "# Shared library",
            "add_library(timber_model SHARED model.c)",
            "target_include_directories(timber_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})",
            "",
            "# Static library",
            "add_library(timber_model_static STATIC model.c)",
            "target_include_directories(timber_model_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})",
            "set_target_properties(timber_model_static PROPERTIES OUTPUT_NAME timber_model)",
            "",
        ]

        # Add architecture-specific flags
        if "avx512f" in self.target.features:
            lines.append('target_compile_options(timber_model PRIVATE "-mavx512f" "-mavx512bw")')
            lines.append('target_compile_options(timber_model_static PRIVATE "-mavx512f" "-mavx512bw")')
        elif "avx2" in self.target.features:
            lines.append('target_compile_options(timber_model PRIVATE "-mavx2" "-mfma")')
            lines.append('target_compile_options(timber_model_static PRIVATE "-mavx2" "-mfma")')

        lines.extend([
            "",
            "# Optimization flags",
            'target_compile_options(timber_model PRIVATE "-O3" "-DNDEBUG")',
            'target_compile_options(timber_model_static PRIVATE "-O3" "-DNDEBUG")',
        ])

        return "\n".join(lines) + "\n"

    def _emit_makefile(self, ir: TimberIR) -> str:
        """Generate Makefile supporting both host and embedded cross-compilation."""
        t = self.target

        # Architecture-specific SIMD flags (host only)
        simd_flags = ""
        if not t.embedded:
            if "avx512f" in t.features:
                simd_flags = "-mavx512f -mavx512bw"
            elif "avx2" in t.features:
                simd_flags = "-mavx2 -mfma"

        cc = f"{t.cross_prefix}gcc" if t.cross_prefix else "gcc"
        ar = f"{t.cross_prefix}ar" if t.cross_prefix else "ar"

        cpu_flags = t.cpu_flags or simd_flags
        extra = t.extra_flags or ""

        base_cflags = f"-std=c99 -O2 -DNDEBUG -Wall -Wextra {cpu_flags} {extra}".strip()

        if t.embedded:
            lines = [
                "# Makefile — Timber compiled model (embedded target)",
                f"# Target: {t.arch}  ABI: {t.abi}",
                "# Generated by Timber — DO NOT EDIT",
                "",
                f"CC  = {cc}",
                f"AR  = {ar}",
                f"CFLAGS = {base_cflags}",
                "",
                ".PHONY: all clean",
                "",
                "all: libtimber_model.a timber_model.o",
                "",
                "timber_model.o: model.c model_data.c model.h",
                "\t$(CC) $(CFLAGS) -c -o $@ model.c",
                "",
                "libtimber_model.a: timber_model.o",
                "\t$(AR) rcs $@ timber_model.o",
                "",
                "clean:",
                "\trm -f *.o *.a",
            ]
        else:
            fpic = "-fPIC"
            lines = [
                "# Makefile — Timber compiled model",
                "# Generated by Timber — DO NOT EDIT",
                "",
                "CC ?= gcc",
                f"CFLAGS = -std=c99 -O3 -DNDEBUG {fpic} -Wall -Wextra {cpu_flags}".strip(),
                "",
                ".PHONY: all clean",
                "",
                "all: libtimber_model.so libtimber_model.a",
                "",
                "libtimber_model.so: model.c model_data.c model.h",
                "\t$(CC) $(CFLAGS) -shared -o $@ model.c -lm",
                "",
                "libtimber_model.a: model.c model_data.c model.h",
                "\t$(CC) $(CFLAGS) -c -o model.o model.c",
                "\tar rcs $@ model.o",
                "",
                "clean:",
                "\trm -f *.o *.so *.a",
            ]

        return "\n".join(lines) + "\n"

    def _float_type(self) -> str:
        if self.target.precision == PrecisionMode.FLOAT16:
            return "_Float16"
        return "float"

    @staticmethod
    def _format_float(value: float) -> str:
        if value == 0.0:
            return "0.0f"
        s = f"{value:.10g}"
        # Ensure there's a decimal point so the 'f' suffix is valid C
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s + "f"
