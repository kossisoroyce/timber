"""RISC-V Vector (RVV) emitter — dynamic-length tree traversal with vsetvl."""

from __future__ import annotations

from timber.codegen.c99 import TargetSpec
from timber.ir.model import TimberIR, TreeEnsembleStage

from timber.accel.accel.simd.base import SIMDEmitterBase


class RVVEmitter(SIMDEmitterBase):
    """Emit RISC-V Vector-extension inference code (dynamic-width float32).

    RVV uses ``vsetvl`` to configure the hardware vector length at runtime,
    similar to ARM SVE but with explicit length negotiation.  The generated
    code uses LMUL=1 (``m1``) grouping by default.
    """

    # Minimum guaranteed vector width for RVV (128 bits in RV32/64).
    _MIN_VECTOR_WIDTH = 128

    def __init__(self, target: TargetSpec, simd_config: dict):
        super().__init__(target, simd_config)
        self._lmul = simd_config.get("lmul", 1)
        self._unroll = simd_config.get("unroll_factor", 2)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def instruction_set_name(self) -> str:
        return "rvv"

    def vector_width_bits(self) -> int:
        return self._MIN_VECTOR_WIDTH

    def compiler_flags(self) -> str:
        return "-march=rv64gcv -mabi=lp64d"

    def emit_simd_includes(self) -> str:
        return (
            "#include <riscv_vector.h>\n"
            "#include <stdint.h>\n"
            "#include <string.h>\n"
        )

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def emit_simd_traversal(self, ir: TimberIR) -> str:
        """Generate RISC-V Vector tree-ensemble traversal."""
        parts: list[str] = []
        parts.append(self._emit_scalar_fallback(ir))
        parts.append(self._emit_simd_infer(ir))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ensemble(self, ir: TimberIR) -> TreeEnsembleStage:
        for stage in ir.pipeline:
            if isinstance(stage, TreeEnsembleStage):
                return stage
        raise ValueError("TimberIR contains no TreeEnsembleStage")

    def _emit_scalar_fallback(self, ir: TimberIR) -> str:
        """Scalar fallback for non-RVV builds."""
        lines: list[str] = []
        lines.append("/* Scalar fallback for non-RVV builds */")
        lines.append("static inline float _timber_scalar_traverse(")
        lines.append("    const float* features,")
        lines.append("    const int32_t* feature_indices,")
        lines.append("    const float* thresholds,")
        lines.append("    const int32_t* left_children,")
        lines.append("    const int32_t* right_children,")
        lines.append("    const int8_t*  is_leaf,")
        lines.append("    const float* leaf_values,")
        lines.append("    int          root,")
        lines.append("    int          n_nodes)")
        lines.append("{")
        lines.append("    int node = root;")
        lines.append("    while (node >= 0 && node < n_nodes && !is_leaf[node]) {")
        lines.append("        float fval = features[feature_indices[node]];")
        lines.append("        node = (fval <= thresholds[node])")
        lines.append("             ? left_children[node]")
        lines.append("             : right_children[node];")
        lines.append("    }")
        lines.append("    return (node >= 0 && node < n_nodes) ? leaf_values[node] : 0.0f;")
        lines.append("}")
        lines.append("")
        lines.append("static void _timber_scalar_infer_batch(")
        lines.append("    const float* inputs, int n_samples, int n_features,")
        lines.append("    float* outputs, int start,")
        lines.append("    const int32_t* feature_indices,")
        lines.append("    const float* thresholds,")
        lines.append("    const int32_t* left_children,")
        lines.append("    const int32_t* right_children,")
        lines.append("    const int8_t*  is_leaf,")
        lines.append("    const float* leaf_values,")
        lines.append("    int n_trees, const int* tree_roots,")
        lines.append("    float base_score, int n_nodes)")
        lines.append("{")
        lines.append("    for (int s = start; s < n_samples; s++) {")
        lines.append("        const float* feat = inputs + s * n_features;")
        lines.append("        float acc = base_score;")
        lines.append("        for (int t = 0; t < n_trees; t++) {")
        lines.append("            acc += _timber_scalar_traverse(")
        lines.append("                feat, feature_indices, thresholds,")
        lines.append("                left_children, right_children,")
        lines.append("                is_leaf, leaf_values, tree_roots[t], n_nodes);")
        lines.append("        }")
        lines.append("        outputs[s] = acc;")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _emit_simd_infer(self, ir: TimberIR) -> str:
        """Emit the main RVV inference function.

        RVV uses ``vsetvl_e32m1`` to negotiate the hardware vector length
        for 32-bit float elements with LMUL=1.  The main loop processes
        ``vl`` samples per iteration; the final iteration naturally handles
        the tail because ``vsetvl`` clamps to the remaining element count.
        """
        ens = self._get_ensemble(ir)
        lmul_suffix = f"m{self._lmul}"
        # Mask type depends on LMUL: m1 -> b32, m2 -> b16, m4 -> b8
        mask_width = 32 // self._lmul
        mask_suffix = f"b{mask_width}"

        lines: list[str] = []

        lines.append("/*")
        lines.append(" * RISC-V Vector (RVV)-accelerated inference")
        lines.append(f" * Trees: {len(ens.trees)}, Features: {ens.n_features}")
        lines.append(f" * LMUL={self._lmul}, element type: float32")
        lines.append(" * Vector length negotiated at runtime via vsetvl.")
        lines.append(" */")
        lines.append("#if defined(__riscv_vector)")
        lines.append("")

        # ---------- per-tree lane traversal ----------
        lines.append("/*")
        lines.append(" * Traverse one tree for `vl` lanes.")
        lines.append(" * Because RVV lacks efficient indexed-gather on node arrays,")
        lines.append(" * the inner traversal is scalar per lane; accumulation is vectorized.")
        lines.append(" */")
        lines.append("static inline void _timber_rvv_traverse_tree_lanes(")
        lines.append("    const float*  inputs,")
        lines.append("    int           n_features,")
        lines.append("    int           s_start,")
        lines.append("    size_t        vl,")
        lines.append("    const int32_t*  feature_indices,")
        lines.append("    const float*  thresholds,")
        lines.append("    const int32_t*  left_children,")
        lines.append("    const int32_t*  right_children,")
        lines.append("    const int8_t*   is_leaf,")
        lines.append("    const float*  leaf_values,")
        lines.append("    int           root,")
        lines.append("    float*        lane_results,")
        lines.append("    int           n_nodes)")
        lines.append("{")
        lines.append("    for (size_t lane = 0; lane < vl; lane++) {")
        lines.append("        const float* feat = inputs + (s_start + (int)lane) * n_features;")
        lines.append("        int node = root;")
        lines.append("        while (node >= 0 && node < n_nodes && !is_leaf[node]) {")
        lines.append("            float fval = feat[feature_indices[node]];")
        lines.append("            node = (fval <= thresholds[node])")
        lines.append("                 ? left_children[node]")
        lines.append("                 : right_children[node];")
        lines.append("        }")
        lines.append("        lane_results[lane] = (node >= 0 && node < n_nodes) ? leaf_values[node] : 0.0f;")
        lines.append("    }")
        lines.append("}")
        lines.append("")

        # ---------- main inference entry point ----------
        lines.append("int timber_infer_simd(")
        lines.append("    const float*     inputs,")
        lines.append("    int              n_samples,")
        lines.append("    float*           outputs,")
        lines.append("    const TimberCtx* ctx)")
        lines.append("{")
        lines.append("    const int n_features      = ctx->n_features;")
        lines.append("    const int n_trees          = ctx->n_trees;")
        lines.append("    const int32_t* feature_indices = ctx->feature_indices;")
        lines.append("    const float* thresholds    = ctx->thresholds;")
        lines.append("    const int32_t* left_children   = ctx->left_children;")
        lines.append("    const int32_t* right_children  = ctx->right_children;")
        lines.append("    const int8_t* is_leaf       = ctx->is_leaf;")
        lines.append("    const float* leaf_values   = ctx->leaf_values;")
        lines.append("    const int* tree_roots       = ctx->tree_roots;")
        lines.append("    const float base_score      = ctx->base_score;")
        lines.append("    const int n_nodes            = ctx->n_nodes;")
        lines.append("")
        lines.append("    /*")
        lines.append("     * RVV strip-mining loop.")
        lines.append(f"     * vsetvl_e32{lmul_suffix} negotiates the vector length for")
        lines.append("     * 32-bit floats with the chosen LMUL grouping.")
        lines.append("     */")
        lines.append("    int s = 0;")
        lines.append("    while (s < n_samples) {")
        lines.append(f"        size_t vl = __riscv_vsetvl_e32{lmul_suffix}((size_t)(n_samples - s));")
        lines.append("")
        lines.append("        /* Temporary buffer sized to current vl (stack VLA) */")
        lines.append("        /* Cap VLA to prevent stack overflow (max 2048 lanes) */")
        lines.append("        const size_t capped_vl = (vl > 2048u) ? 2048u : vl;")
        lines.append("        float tree_buf[capped_vl];")
        lines.append("")
        lines.append("        /* Initialize accumulator with base_score */")
        lines.append(f"        vfloat32{lmul_suffix}_t vacc = __riscv_vfmv_v_f_f32{lmul_suffix}(base_score, vl);")
        lines.append("")
        lines.append("        for (int t = 0; t < n_trees; t++) {")
        lines.append("            _timber_rvv_traverse_tree_lanes(")
        lines.append("                inputs, n_features, s, vl,")
        lines.append("                feature_indices, thresholds,")
        lines.append("                left_children, right_children,")
        lines.append("                is_leaf, leaf_values, tree_roots[t],")
        lines.append("                tree_buf, n_nodes);")
        lines.append("")
        lines.append("            /* Load lane results and accumulate */")
        lines.append(f"            vfloat32{lmul_suffix}_t vtree = __riscv_vle32_v_f32{lmul_suffix}(tree_buf, vl);")
        lines.append(f"            vacc = __riscv_vfadd_vv_f32{lmul_suffix}(vacc, vtree, vl);")
        lines.append("        }")
        lines.append("")
        lines.append("        /* Store results for this strip */")
        lines.append(f"        __riscv_vse32_v_f32{lmul_suffix}(outputs + s, vacc, vl);")
        lines.append("")
        lines.append("        s += (int)vl;")
        lines.append("    }")
        lines.append("")
        lines.append("    return 0;")
        lines.append("}")
        lines.append("")
        lines.append("#else")
        lines.append("/* RISC-V Vector not available — fall back to scalar */")
        lines.append("int timber_infer_simd(")
        lines.append("    const float*     inputs,")
        lines.append("    int              n_samples,")
        lines.append("    float*           outputs,")
        lines.append("    const TimberCtx* ctx)")
        lines.append("{")
        lines.append("    _timber_scalar_infer_batch(")
        lines.append("        inputs, n_samples, ctx->n_features,")
        lines.append("        outputs, 0,")
        lines.append("        ctx->feature_indices, ctx->thresholds,")
        lines.append("        ctx->left_children, ctx->right_children,")
        lines.append("        ctx->is_leaf, ctx->leaf_values,")
        lines.append("        ctx->n_trees, ctx->tree_roots, ctx->base_score, ctx->n_nodes);")
        lines.append("    return 0;")
        lines.append("}")
        lines.append("#endif /* __riscv_vector */")
        lines.append("")
        return "\n".join(lines)
