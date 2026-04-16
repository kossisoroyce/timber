"""ARM SVE SIMD emitter — scalable-vector tree traversal with predicated operations."""

from __future__ import annotations

from timber.accel.accel.simd.base import SIMDEmitterBase
from timber.codegen.c99 import TargetSpec
from timber.ir.model import TimberIR, TreeEnsembleStage


class SVEEmitter(SIMDEmitterBase):
    """Emit ARM SVE-vectorized inference code (scalable-width float32).

    SVE differs from fixed-width SIMD in that the vector length is determined
    at runtime via ``svcntw()``.  All operations are predicated with
    ``svbool_t`` masks, making the generated code naturally handle arbitrary
    vector lengths without a separate scalar remainder loop.
    """

    # SVE vector length is runtime-determined; we report the minimum
    # guaranteed width (128 bits) for the header define.
    _MIN_VECTOR_WIDTH = 128

    def __init__(self, target: TargetSpec, simd_config: dict):
        super().__init__(target, simd_config)
        self._unroll = simd_config.get("unroll_factor", 2)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def instruction_set_name(self) -> str:
        return "sve"

    def vector_width_bits(self) -> int:
        return self._MIN_VECTOR_WIDTH

    def compiler_flags(self) -> str:
        return "-march=armv8-a+sve"

    def emit_simd_includes(self) -> str:
        return (
            "#include <arm_sve.h>\n"
            "#include <stdint.h>\n"
            "#include <string.h>\n"
        )

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def emit_simd_traversal(self, ir: TimberIR) -> str:
        """Generate ARM SVE vectorized tree-ensemble traversal."""
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
        """Scalar fallback used when SVE is unavailable at compile time."""
        lines: list[str] = []
        lines.append("/* Scalar fallback for non-SVE builds */")
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
        """Emit the main ARM SVE inference function.

        SVE's predicated design means the main loop uses ``svwhilelt``
        to generate a lane predicate that naturally masks off inactive
        lanes in the final iteration — no separate scalar tail is needed
        at the SVE level.
        """
        ens = self._get_ensemble(ir)
        lines: list[str] = []

        lines.append("/*")
        lines.append(" * ARM SVE-accelerated inference — scalable-width float32")
        lines.append(f" * Trees: {len(ens.trees)}, Features: {ens.n_features}")
        lines.append(" * Vector length is determined at runtime via svcntw().")
        lines.append(" * All operations are predicated; no scalar remainder needed.")
        lines.append(" */")
        lines.append("#if defined(__ARM_FEATURE_SVE)")
        lines.append("")

        # ---------- per-tree traversal (scalar per-lane, SVE accumulate) ----------
        lines.append("/*")
        lines.append(" * Traverse one tree for VL lanes simultaneously.")
        lines.append(" * We gather features per-lane, compare, and follow children.")
        lines.append(" * Because SVE lacks efficient per-lane gather on node indices,")
        lines.append(" * the inner traversal is scalar per lane; the accumulation is SVE.")
        lines.append(" */")
        lines.append("static inline void _timber_sve_traverse_tree_lanes(")
        lines.append("    const float* inputs,")
        lines.append("    int           n_features,")
        lines.append("    int           s_start,")
        lines.append("    int           lane_count,")
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
        lines.append("    for (int lane = 0; lane < lane_count; lane++) {")
        lines.append("        const float* feat = inputs + (s_start + lane) * n_features;")
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
        lines.append("    /* Runtime vector length in 32-bit elements */")
        lines.append("    const uint64_t vl = svcntw();")
        lines.append("")
        lines.append("    /* Temporary buffer for per-tree lane results (stack VLA) */")
        lines.append("    /* Cap VLA to prevent stack overflow (max 2048 lanes) */")
        lines.append("    const uint64_t capped_vl = (vl > 2048u) ? 2048u : vl;")
        lines.append("    float tree_buf[capped_vl];")
        lines.append("")
        lines.append("    /* SVE loop with predicated tail handling via svwhilelt */")
        lines.append("    int s = 0;")
        lines.append("    do {")
        lines.append("        svbool_t pg = svwhilelt_b32_s32(s, n_samples);")
        lines.append("        int lane_count = (int)((n_samples - s) < (int)capped_vl")
        lines.append("                              ? (n_samples - s) : (int)capped_vl);")
        lines.append("")
        lines.append("        /* Broadcast base_score into accumulator */")
        lines.append("        svfloat32_t vacc = svdup_n_f32(base_score);")
        lines.append("")
        lines.append("        for (int t = 0; t < n_trees; t++) {")
        lines.append("            _timber_sve_traverse_tree_lanes(")
        lines.append("                inputs, n_features, s, lane_count,")
        lines.append("                feature_indices, thresholds,")
        lines.append("                left_children, right_children,")
        lines.append("                is_leaf, leaf_values, tree_roots[t],")
        lines.append("                tree_buf, n_nodes);")
        lines.append("")
        lines.append("            /* Load per-lane tree results and accumulate */")
        lines.append("            svfloat32_t vtree = svld1_f32(pg, tree_buf);")
        lines.append("            vacc = svadd_f32_m(pg, vacc, vtree);")
        lines.append("        }")
        lines.append("")
        lines.append("        /* Store results (predicated — only active lanes written) */")
        lines.append("        svst1_f32(pg, outputs + s, vacc);")
        lines.append("")
        lines.append("        s += (int)vl;")
        lines.append("    } while (s < n_samples);")
        lines.append("")
        lines.append("    return 0;")
        lines.append("}")
        lines.append("")
        lines.append("#else")
        lines.append("/* ARM SVE not available — fall back to scalar */")
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
        lines.append("#endif /* __ARM_FEATURE_SVE */")
        lines.append("")
        return "\n".join(lines)
