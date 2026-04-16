"""Base SIMD emitter and factory."""

from __future__ import annotations

import abc
import re
import warnings
from dataclasses import dataclass

from timber.codegen.c99 import C99Emitter, C99Output, TargetSpec
from timber.ir.model import TimberIR, TreeEnsembleStage


class SIMDEmitterBase(abc.ABC):
    """Abstract base for SIMD-specialized code emitters."""

    def __init__(self, target: TargetSpec, simd_config: dict):
        self.target = target
        self.simd_config = simd_config
        self._c99 = C99Emitter(target=target)

    @abc.abstractmethod
    def instruction_set_name(self) -> str:
        ...

    @abc.abstractmethod
    def vector_width_bits(self) -> int:
        ...

    @abc.abstractmethod
    def emit_simd_traversal(self, ir: TimberIR) -> str:
        """Generate SIMD-vectorized tree traversal code."""
        ...

    @abc.abstractmethod
    def emit_simd_includes(self) -> str:
        """Return architecture-specific includes."""
        ...

    @abc.abstractmethod
    def compiler_flags(self) -> str:
        """Return required compiler flags."""
        ...

    def emit(self, ir: TimberIR) -> C99Output:
        """Generate SIMD-accelerated C code."""
        # Get baseline C99 output
        baseline = self._c99.emit(ir)

        # Generate SIMD traversal code
        simd_code = self.emit_simd_traversal(ir)
        simd_includes = self.emit_simd_includes()
        flags = self.compiler_flags()

        # Expand TimberCtx and generate flat arrays for SIMD access
        model_c = self._expand_ctx_for_simd(baseline.model_c, ir)
        model_data_c = self._generate_flat_arrays(baseline.model_data_c, ir)

        # Inject SIMD code into model.c
        model_c = self._inject_simd(model_c, simd_includes, simd_code)

        # Update header with SIMD defines
        model_h = self._inject_simd_defines(baseline.model_h)

        # Update build files with SIMD flags
        cmakelists = self._patch_cmake(baseline.cmakelists, flags)
        makefile = self._patch_makefile(baseline.makefile, flags)

        return C99Output(
            model_c=model_c,
            model_h=model_h,
            model_data_c=model_data_c,
            cmakelists=cmakelists,
            makefile=makefile,
        )

    @staticmethod
    def _find_function_range(lines: list[str], func_name: str) -> tuple[int, int] | None:
        """Find the line range [start, end] of a C function definition by brace-counting.

        Returns (start_line_index, end_line_index) inclusive, or None if not found.
        """
        pattern = re.compile(r'\s*int\s+' + re.escape(func_name) + r'\b\s*\(')
        start = None
        for i, line in enumerate(lines):
            if pattern.match(line):
                start = i
                break
        if start is None:
            return None

        depth = 0
        found_open = False
        for i in range(start, len(lines)):
            for ch in lines[i]:
                if ch == '{':
                    depth += 1
                    found_open = True
                elif ch == '}':
                    depth -= 1
                    if found_open and depth == 0:
                        return (start, i)
        return None

    @staticmethod
    def _get_ensemble_stage(ir: TimberIR) -> TreeEnsembleStage:
        """Extract the first TreeEnsembleStage from the IR."""
        for stage in ir.pipeline:
            if isinstance(stage, TreeEnsembleStage):
                return stage
        raise ValueError("TimberIR contains no TreeEnsembleStage")

    def _expand_ctx_for_simd(self, model_c: str, ir: TimberIR) -> str:
        """Expand the minimal TimberCtx struct to include SIMD-required fields.

        The baseline C99 emitter generates ``struct TimberCtx { int initialized; };``
        which is insufficient for SIMD emitters that access ``ctx->n_features``,
        ``ctx->feature_indices``, etc.  This method replaces the struct definition
        and the default context initialisation with a version that carries all the
        flat tree-array pointers the SIMD traversal code needs.
        """
        ens = self._get_ensemble_stage(ir)
        n_trees = len(ens.trees)
        total_nodes = sum(len(t.nodes) for t in ens.trees)

        # --- Replace struct TimberCtx ---
        old_struct = "struct TimberCtx {\n    int initialized;\n};"
        new_struct = (
            "struct TimberCtx {\n"
            "    int initialized;\n"
            "    /* SIMD-accelerated fields */\n"
            "    int n_features;\n"
            "    int n_trees;\n"
            "    int n_nodes;\n"
            "    const int32_t* feature_indices;\n"
            "    const float*   thresholds;\n"
            "    const int32_t* left_children;\n"
            "    const int32_t* right_children;\n"
            "    const int8_t*  is_leaf;\n"
            "    const float*   leaf_values;\n"
            "    const int*     tree_roots;\n"
            "    float          base_score;\n"
            "};"
        )
        model_c = model_c.replace(old_struct, new_struct)

        # --- Replace _default_ctx initialisation ---
        old_default = "static struct TimberCtx _default_ctx = {1};"
        new_default = (
            "static struct TimberCtx _default_ctx = {\n"
            "    .initialized = 0  /* populated by timber_init */\n"
            "};"
        )
        model_c = model_c.replace(old_default, new_default)

        # --- Expand timber_init to populate flat-array pointers ---
        # Find and replace the timber_init body
        lines = model_c.split("\n")
        init_range = self._find_function_range(lines, "timber_init")
        if init_range is not None:
            start, end = init_range
            init_body = [
                "int timber_init(TimberCtx** ctx) {",
                "    if (ctx == NULL) {",
                '        timber_log(0, "timber_init: ctx is NULL");',
                "        return TIMBER_ERR_NULL;",
                "    }",
                "    _default_ctx.initialized     = 1;",
                f"    _default_ctx.n_features      = TIMBER_N_FEATURES;",
                f"    _default_ctx.n_trees         = {n_trees};",
                f"    _default_ctx.n_nodes         = {total_nodes};",
                "    _default_ctx.feature_indices  = _timber_flat_features;",
                "    _default_ctx.thresholds       = _timber_flat_thresholds;",
                "    _default_ctx.left_children    = _timber_flat_left;",
                "    _default_ctx.right_children   = _timber_flat_right;",
                "    _default_ctx.is_leaf          = _timber_flat_is_leaf;",
                "    _default_ctx.leaf_values      = _timber_flat_leaves;",
                "    _default_ctx.tree_roots       = _timber_flat_tree_roots;",
                "    _default_ctx.base_score       = TIMBER_BASE_SCORE;",
                "    *ctx = &_default_ctx;",
                '    timber_log(2, "timber_init: OK");',
                "    return TIMBER_OK;",
                "}",
            ]
            lines[start:end + 1] = init_body
            model_c = "\n".join(lines)

        return model_c

    def _generate_flat_arrays(self, model_data_c: str, ir: TimberIR) -> str:
        """Append flattened tree arrays to model_data.c for SIMD ctx-> access.

        Combines the per-tree static arrays (tree_0_features, tree_1_features, ...)
        into contiguous flat arrays plus a tree_roots index.
        """
        ens = self._get_ensemble_stage(ir)
        n_trees = len(ens.trees)

        # Build flat arrays from the IR (same source of truth as per-tree arrays)
        flat_features: list[int] = []
        flat_thresholds: list[float] = []
        flat_left: list[int] = []
        flat_right: list[int] = []
        flat_is_leaf: list[int] = []
        flat_leaves: list[float] = []
        tree_roots: list[int] = []

        for tree in ens.trees:
            tree_roots.append(len(flat_features))
            for node in tree.nodes:
                flat_features.append(node.feature_index if not node.is_leaf else -1)
                flat_thresholds.append(node.threshold if not node.is_leaf else 0.0)
                flat_left.append(node.left_child if not node.is_leaf else -1)
                flat_right.append(node.right_child if not node.is_leaf else -1)
                flat_is_leaf.append(1 if node.is_leaf else 0)
                flat_leaves.append(node.leaf_value if node.is_leaf else 0.0)

        total_nodes = len(flat_features)

        def _c_int_array(name: str, values: list[int]) -> str:
            body = ", ".join(str(v) for v in values)
            return f"static const int32_t {name}[{len(values)}] = {{{body}}};"

        def _c_int_array_plain(name: str, values: list[int]) -> str:
            body = ", ".join(str(v) for v in values)
            return f"static const int {name}[{len(values)}] = {{{body}}};"

        def _c_float_array(name: str, values: list[float]) -> str:
            body = ", ".join(f"{v}f" for v in values)
            return f"static const float {name}[{len(values)}] = {{{body}}};"

        def _c_int8_array(name: str, values: list[int]) -> str:
            body = ", ".join(str(v) for v in values)
            return f"static const int8_t {name}[{len(values)}] = {{{body}}};"

        flat_code = [
            "",
            f"/* === Flat tree arrays for SIMD access ({total_nodes} total nodes, {n_trees} trees) === */",
            _c_int_array("_timber_flat_features", flat_features),
            _c_float_array("_timber_flat_thresholds", flat_thresholds),
            _c_int_array("_timber_flat_left", flat_left),
            _c_int_array("_timber_flat_right", flat_right),
            _c_int8_array("_timber_flat_is_leaf", flat_is_leaf),
            _c_float_array("_timber_flat_leaves", flat_leaves),
            _c_int_array_plain("_timber_flat_tree_roots", tree_roots),
            "",
        ]

        return model_data_c + "\n".join(flat_code)

    def _inject_simd(self, model_c: str, includes: str, simd_code: str) -> str:
        lines = model_c.split("\n")

        # Insert includes after existing includes
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("#include"):
                insert_idx = i + 1
        lines.insert(insert_idx, includes)

        # Detect float type from the source
        float_type = "float"
        for line in lines:
            if "_Float16" in line and ("inputs" in line or "outputs" in line):
                float_type = "_Float16"
                break

        # Replace timber_infer_single with SIMD delegation
        range_single = self._find_function_range(lines, "timber_infer_single")
        if range_single is not None:
            start, end = range_single
            replacement = [
                f"int timber_infer_single(",
                f"    const {float_type}  inputs[TIMBER_N_FEATURES],",
                f"    {float_type}        outputs[TIMBER_N_OUTPUTS],",
                f"    const TimberCtx*    ctx",
                f") {{",
                f"    return timber_infer_simd(inputs, 1, outputs, ctx);",
                f"}}",
            ]
            lines[start:end + 1] = replacement
        else:
            warnings.warn(
                "Could not locate 'timber_infer_single' in generated C99 — "
                "SIMD injection skipped. The upstream C99 emitter may have changed.",
                RuntimeWarning,
                stacklevel=3,
            )

        # Replace timber_infer (batch) with SIMD delegation
        range_batch = self._find_function_range(lines, "timber_infer")
        # Ensure we don't re-match timber_infer_single or timber_infer_simd
        if range_batch is not None:
            sig_text = " ".join(lines[range_batch[0]:range_batch[0] + 3])
            if "timber_infer_single" in sig_text or "timber_infer_simd" in sig_text:
                # Search again starting after this match
                remaining = lines[range_batch[1] + 1:]
                range_batch2 = self._find_function_range(remaining, "timber_infer")
                if range_batch2 is not None:
                    offset = range_batch[1] + 1
                    range_batch = (range_batch2[0] + offset, range_batch2[1] + offset)
                else:
                    range_batch = None

        if range_batch is not None:
            start, end = range_batch
            replacement = [
                f"int timber_infer(",
                f"    const {float_type}*  inputs,",
                f"    int                  n_samples,",
                f"    {float_type}*        outputs,",
                f"    const TimberCtx*     ctx",
                f") {{",
                f"    if (inputs == NULL || outputs == NULL) return TIMBER_ERR_NULL;",
                f"    if (n_samples <= 0) return TIMBER_ERR_BOUNDS;",
                f"    return timber_infer_simd(inputs, n_samples, outputs, ctx);",
                f"}}",
            ]
            lines[start:end + 1] = replacement
        else:
            warnings.warn(
                "Could not locate 'timber_infer' (batch) in generated C99 — "
                "SIMD injection skipped. The upstream C99 emitter may have changed.",
                RuntimeWarning,
                stacklevel=3,
            )

        result = "\n".join(lines)

        # Forward-declare timber_infer_simd before the replaced functions
        fwd_decl = "\n/* Forward declaration for SIMD inference */\n"
        fwd_decl += (
            f"int timber_infer_simd(const {float_type}* inputs, int n_samples, "
            f"{float_type}* outputs, const TimberCtx* ctx);\n"
        )

        idx = result.find("int timber_infer_single(")
        if idx >= 0:
            result = result[:idx] + fwd_decl + "\n" + result[idx:]

        # Append SIMD functions at the end
        result += "\n\n/* === SIMD-Accelerated Inference === */\n"
        result += simd_code
        return result

    def _inject_simd_defines(self, model_h: str) -> str:
        define = f"#define TIMBER_ACCEL_SIMD \"{self.instruction_set_name()}\"\n"
        define += f"#define TIMBER_ACCEL_VECTOR_WIDTH {self.vector_width_bits()}\n"
        # Insert after existing defines
        lines = model_h.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("#define TIMBER_N_"):
                lines.insert(i, define)
                break
        # Detect float type from header
        float_type = "float"
        for line in lines:
            if "_Float16" in line and ("inputs" in line or "outputs" in line):
                float_type = "_Float16"
                break
        # Add SIMD infer declaration
        lines.append("")
        lines.append(f"/* SIMD-accelerated inference ({self.instruction_set_name()}) */")
        lines.append(
            f"int timber_infer_simd(const {float_type}* inputs, int n_samples, "
            f"{float_type}* outputs, const TimberCtx* ctx);"
        )
        return "\n".join(lines)

    def _patch_cmake(self, cmake: str, flags: str) -> str:
        return cmake.replace(
            "set(CMAKE_C_FLAGS",
            f"set(CMAKE_C_FLAGS \"{flags} ${{CMAKE_C_FLAGS}}\")\n# Original:\n# set(CMAKE_C_FLAGS",
        ) if "set(CMAKE_C_FLAGS" in cmake else cmake + f"\nset(CMAKE_C_FLAGS \"${{CMAKE_C_FLAGS}} {flags}\")\n"

    def _patch_makefile(self, makefile: str, flags: str) -> str:
        return makefile.replace("CFLAGS =", f"CFLAGS = {flags}") if "CFLAGS =" in makefile else f"CFLAGS += {flags}\n" + makefile


def get_simd_emitter(profile) -> SIMDEmitterBase:
    """Factory: return the correct SIMD emitter for a target profile."""
    from timber.accel.accel.simd.avx2 import AVX2Emitter
    from timber.accel.accel.simd.avx512 import AVX512Emitter
    from timber.accel.accel.simd.neon import NEONEmitter
    from timber.accel.accel.simd.sve import SVEEmitter
    from timber.accel.accel.simd.rvv import RVVEmitter

    isa = profile.simd_config.get("instruction_set", "")
    mapping = {
        "avx2": AVX2Emitter,
        "avx512": AVX512Emitter,
        "neon": NEONEmitter,
        "sve": SVEEmitter,
        "rvv": RVVEmitter,
    }
    cls = mapping.get(isa)
    if cls is None:
        raise ValueError(f"Unknown SIMD instruction set: {isa!r}. Supported: {list(mapping)}")
    return cls(target=profile.target_spec, simd_config=profile.simd_config)
