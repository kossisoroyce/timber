"""AVX2 SIMD emitter — 8-wide float tree traversal using 256-bit intrinsics."""

from __future__ import annotations

from timber.codegen.c99 import TargetSpec
from timber.ir.model import TimberIR, TreeEnsembleStage

from timber.accel.accel.simd.base import SIMDEmitterBase


class AVX2Emitter(SIMDEmitterBase):
    """Emit AVX2-vectorized inference code (8-wide float32)."""

    VECTOR_WIDTH = 8  # 256 bits / 32 bits per float

    def __init__(self, target: TargetSpec, simd_config: dict):
        super().__init__(target, simd_config)
        self._unroll = simd_config.get("unroll_factor", 2)
        self._use_fma = simd_config.get("use_fma", True)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def instruction_set_name(self) -> str:
        return "avx2"

    def vector_width_bits(self) -> int:
        return 256

    def compiler_flags(self) -> str:
        flags = "-mavx2"
        if self._use_fma:
            flags += " -mfma"
        return flags

    def emit_simd_includes(self) -> str:
        return (
            "#include <immintrin.h>\n"
            "#include <stdint.h>\n"
            "#include <string.h>\n"
        )

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def emit_simd_traversal(self, ir: TimberIR) -> str:
        """Generate AVX2 vectorized tree-ensemble traversal."""
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
        """Scalar fallback for remainder samples."""
        lines: list[str] = []
        lines.append("/* Scalar fallback for remainder elements */")
        lines.append("static inline float _timber_scalar_traverse(")
        lines.append("    const float* features,")
        lines.append("    const int32_t* feature_indices,")
        lines.append("    const float* thresholds,")
        lines.append("    const int32_t* left_children,")
        lines.append("    const int32_t* right_children,")
        lines.append("    const int8_t*  is_leaf,")
        lines.append("    const float* leaf_values,")
        lines.append("    int          root,")
        lines.append("    int          n_nodes,")
        lines.append("    int          n_features)")
        lines.append("{")
        lines.append("    int node = root;")
        lines.append("    while (node >= 0 && node < n_nodes && !is_leaf[node]) {")
        lines.append("        int fid = feature_indices[node];")
        lines.append("        if (fid < 0 || fid >= n_features) fid = 0;  /* safety clamp */")
        lines.append("        float fval = features[fid];")
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
        lines.append("                is_leaf, leaf_values, tree_roots[t],")
        lines.append("                n_nodes, n_features);")
        lines.append("        }")
        lines.append("        outputs[s] = acc;")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _emit_simd_infer(self, ir: TimberIR) -> str:
        """Emit the main AVX2 SIMD inference function.

        Uses ctx-> fields at runtime for all model data — no compile-time
        static arrays or defines needed.
        """
        ens = self._get_ensemble(ir)
        W = self.VECTOR_WIDTH
        n_trees = len(ens.trees)
        lines: list[str] = []

        lines.append("/*")
        lines.append(f" * AVX2-accelerated inference — {W}-wide float32 SIMD")
        lines.append(f" * Trees: {n_trees}, Features: {ens.n_features}")
        lines.append(" */")
        lines.append("#ifdef __AVX2__")
        lines.append("")

        # ---------- single-tree vectorized traversal ----------
        lines.append("static inline __m256 _timber_avx2_traverse_tree(")
        lines.append("    const float* feature_ptrs[8],")
        lines.append("    const int32_t* feature_indices,")
        lines.append("    const float* thresholds,")
        lines.append("    const int32_t* left_children,")
        lines.append("    const int32_t* right_children,")
        lines.append("    const int8_t*  is_leaf,")
        lines.append("    const float* leaf_values,")
        lines.append("    int          root,")
        lines.append("    int          n_nodes,")
        lines.append("    int          n_features)")
        lines.append("{")
        lines.append(f"    int nodes[{W}];")
        lines.append(f"    float results[{W}] __attribute__((aligned(32)));")
        lines.append(f"    int active = {W};")
        lines.append("")
        lines.append(f"    for (int i = 0; i < {W}; i++) {{")
        lines.append("        nodes[i] = root;")
        lines.append("        results[i] = 0.0f;")
        lines.append("    }")
        lines.append("")
        lines.append("    while (active > 0) {")
        lines.append("        active = 0;")
        lines.append(f"        float feat_vals[{W}] __attribute__((aligned(32)));")
        lines.append(f"        float thresh_vals[{W}] __attribute__((aligned(32)));")
        lines.append("")
        lines.append(f"        for (int i = 0; i < {W}; i++) {{")
        lines.append("            int n = nodes[i];")
        lines.append("            if (n < 0 || n >= n_nodes || is_leaf[n]) {")
        lines.append("                feat_vals[i] = 0.0f;")
        lines.append("                thresh_vals[i] = 0.0f;")
        lines.append("            } else {")
        lines.append("                int fid = feature_indices[n];")
        lines.append("                if (fid < 0 || fid >= n_features) fid = 0;  /* safety clamp */")
        lines.append("                feat_vals[i] = feature_ptrs[i][fid];")
        lines.append("                thresh_vals[i] = thresholds[n];")
        lines.append("                active++;")
        lines.append("            }")
        lines.append("        }")
        lines.append("")
        lines.append("        if (active == 0) break;")
        lines.append("")
        lines.append("        __m256 vfeat   = _mm256_load_ps(feat_vals);")
        lines.append("        __m256 vthresh = _mm256_load_ps(thresh_vals);")
        lines.append("        __m256 vcmp    = _mm256_cmp_ps(vfeat, vthresh, _CMP_LE_OQ);")
        lines.append("        int mask = _mm256_movemask_ps(vcmp);")
        lines.append("")
        lines.append(f"        for (int i = 0; i < {W}; i++) {{")
        lines.append("            int n = nodes[i];")
        lines.append("            if (n >= 0 && n < n_nodes && !is_leaf[n]) {")
        lines.append("                nodes[i] = (mask & (1u << i))")
        lines.append("                         ? left_children[n]")
        lines.append("                         : right_children[n];")
        lines.append("            }")
        lines.append("        }")
        lines.append("    }")
        lines.append("")
        lines.append(f"    for (int i = 0; i < {W}; i++) {{")
        lines.append("        int n = nodes[i];")
        lines.append("        results[i] = (n >= 0 && n < n_nodes) ? leaf_values[n] : 0.0f;")
        lines.append("    }")
        lines.append("    return _mm256_load_ps(results);")
        lines.append("}")
        lines.append("")

        # ---------- SIMD-accelerated single-sample traversal ----------
        lines.append("static inline float _timber_avx2_traverse_single(")
        lines.append("    const float* input,")
        lines.append("    const int32_t* feature_indices,")
        lines.append("    const float* thresholds,")
        lines.append("    const int32_t* left_children,")
        lines.append("    const int32_t* right_children,")
        lines.append("    const int8_t*  is_leaf,")
        lines.append("    const float* leaf_values,")
        lines.append("    int          root,")
        lines.append("    int          n_nodes,")
        lines.append("    int          n_features)")
        lines.append("{")
        lines.append("    int node = root;")
        lines.append("    while (node >= 0 && node < n_nodes && !is_leaf[node]) {")
        lines.append("        int fid = feature_indices[node];")
        lines.append("        if (fid < 0 || fid >= n_features) fid = 0;  /* safety clamp */")
        lines.append("        float fval = input[fid];")
        lines.append("        node = (fval <= thresholds[node])")
        lines.append("             ? left_children[node] : right_children[node];")
        lines.append("    }")
        lines.append("    return (node >= 0 && node < n_nodes) ? leaf_values[node] : 0.0f;")
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
        lines.append(f"    const int simd_end = n_samples - (n_samples % {W});")
        lines.append("")
        lines.append(f"    for (int s = 0; s < simd_end; s += {W}) {{")
        lines.append("        __m256 vacc = _mm256_set1_ps(base_score);")
        lines.append(f"        const float* feature_ptrs[{W}];")
        lines.append(f"        for (int i = 0; i < {W}; i++) {{")
        lines.append("            feature_ptrs[i] = inputs + (s + i) * n_features;")
        lines.append("        }")
        lines.append("")
        lines.append("        for (int t = 0; t < n_trees; t++) {")
        lines.append("            vacc = _mm256_add_ps(vacc, _timber_avx2_traverse_tree(")
        lines.append("                feature_ptrs, feature_indices, thresholds,")
        lines.append("                left_children, right_children,")
        lines.append("                is_leaf, leaf_values, tree_roots[t],")
        lines.append("                n_nodes, n_features));")
        lines.append("        }")
        lines.append("")
        lines.append("        _mm256_storeu_ps(outputs + s, vacc);")
        lines.append("    }")
        lines.append("")
        lines.append("    /* Scalar fallback for remainder */")
        lines.append("    if (simd_end < n_samples) {")
        lines.append("        _timber_scalar_infer_batch(")
        lines.append("            inputs, n_samples, n_features,")
        lines.append("            outputs, simd_end,")
        lines.append("            feature_indices, thresholds,")
        lines.append("            left_children, right_children,")
        lines.append("            is_leaf, leaf_values,")
        lines.append("            n_trees, tree_roots, base_score, n_nodes);")
        lines.append("    }")
        lines.append("")
        lines.append("    return 0;")
        lines.append("}")

        lines.append("")
        lines.append("#else")
        lines.append("/* AVX2 not available — fall back to scalar */")
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
        lines.append("#endif /* __AVX2__ */")
        lines.append("")
        return "\n".join(lines)
