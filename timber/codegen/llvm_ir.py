"""LLVM IR target backend — emits LLVM IR text format from Timber IR.

Generates human-readable LLVM IR (`.ll` files) that can be compiled with
`llc` or `clang` for hardware-specific optimization on any LLVM-supported
target (x86-64, AArch64, ARM Cortex-M, RISC-V, WebAssembly, etc.).

Supported Timber IR stages:
  - TreeEnsembleStage  → flat basic-block tree traversal in LLVM IR
  - LinearStage        → dot-product + activation in LLVM IR
  - SVMStage           → kernel evaluation in LLVM IR

Output:
  - ``model.ll``     — LLVM IR module with the inference function
  - ``model.bc``     — Binary bitcode (produced by ``llvm-as``, on demand)

Usage::

    from timber.codegen.llvm_ir import LLVMIREmitter, LLVMIROutput
    emitter = LLVMIREmitter(target_triple="aarch64-unknown-linux-gnu")
    result  = emitter.emit(ir)
    print(result.model_ll)

The emitted IR uses SSA form with explicit basic blocks per tree node.
"""

from __future__ import annotations

from dataclasses import dataclass

from timber.ir.model import (
    LinearStage,
    Objective,
    SVMStage,
    TimberIR,
    Tree,
    TreeEnsembleStage,
)

# ---------------------------------------------------------------------------
# Common LLVM triple / datalayout strings for known profiles
# ---------------------------------------------------------------------------
LLVM_TRIPLES: dict[str, tuple[str, str]] = {
    "x86_64":      ("x86_64-unknown-linux-gnu",
                    "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"),
    "aarch64":     ("aarch64-unknown-linux-gnu",
                    "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"),
    "cortex-m4":   ("thumbv7em-none-eabi",
                    "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"),
    "cortex-m33":  ("thumbv8m.main-none-eabi",
                    "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"),
    "rv32imf":     ("riscv32-unknown-elf",
                    "e-m:e-p:32:32-i64:64-n32-S128"),
    "rv64gc":      ("riscv64-unknown-linux-gnu",
                    "e-m:e-p:64:64-i64:64-i128:128-n64-S128"),
    "wasm32":      ("wasm32-unknown-unknown",
                    "e-m:e-p:32:32-i64:64-n32:64-S128"),
}


@dataclass
class LLVMIROutput:
    """Generated LLVM IR module."""
    model_ll: str            # Human-readable LLVM IR text
    target_triple: str       # e.g. "x86_64-unknown-linux-gnu"
    target_datalayout: str   # LLVM data layout string

    def save(self, out_dir) -> dict[str, str]:
        """Write model.ll to out_dir, return dict of {filename: path}."""
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "model.ll"
        p.write_text(self.model_ll, encoding="utf-8")
        return {"model.ll": str(p)}


class LLVMIREmitter:
    """Emit LLVM IR text format from Timber IR.

    Parameters
    ----------
    target_triple:
        LLVM target triple string or a short profile name from
        ``LLVM_TRIPLES`` (e.g. ``"x86_64"``, ``"cortex-m4"``).
    opt_level:
        Optimization level hint embedded as metadata (0–3).
    use_float:
        If True, use ``float`` (f32) precision; otherwise ``double`` (f64).
    """

    def __init__(
        self,
        target_triple: str = "x86_64",
        opt_level: int = 2,
        use_float: bool = True,
    ) -> None:
        if target_triple in LLVM_TRIPLES:
            self.target_triple, self.target_datalayout = LLVM_TRIPLES[target_triple]
        else:
            self.target_triple = target_triple
            self.target_datalayout = ""
        self.opt_level = opt_level
        self.use_float = use_float
        self._fp = "float" if use_float else "double"
        self._fp_suffix = "f" if use_float else ""
        self._reg: int = 0      # SSA register counter (reset per function)

    def emit(self, ir: TimberIR) -> LLVMIROutput:
        """Emit a complete LLVM IR module for the given IR."""
        primary = None
        for stage in ir.pipeline:
            if isinstance(stage, (TreeEnsembleStage, LinearStage, SVMStage)):
                primary = stage
                break
        if primary is None:
            raise ValueError("No supported primary stage in IR pipeline")

        lines: list[str] = []
        lines += self._module_header()

        if isinstance(primary, LinearStage):
            lines += self._emit_linear(primary)
        elif isinstance(primary, SVMStage):
            lines += self._emit_svm(primary)
        else:
            lines += self._emit_tree_ensemble(primary)

        lines += self._module_footer()

        return LLVMIROutput(
            model_ll="\n".join(lines) + "\n",
            target_triple=self.target_triple,
            target_datalayout=self.target_datalayout,
        )

    # ------------------------------------------------------------------
    # Module header / footer
    # ------------------------------------------------------------------

    def _module_header(self) -> list[str]:
        lines = [
            "; ModuleID = 'timber_model'",
            '; source_filename = "timber_model"',
        ]
        if self.target_datalayout:
            lines.append(f'target datalayout = "{self.target_datalayout}"')
        lines.append(f'target triple = "{self.target_triple}"')
        lines += [
            "",
            "; Timber compiled model — LLVM IR",
            "; Generated by Timber — DO NOT EDIT",
            "",
        ]
        return lines

    def _module_footer(self) -> list[str]:
        return [
            "",
            '!llvm.module.flags = !{!0, !1}',
            '!0 = !{i32 1, !"wchar_size", i32 4}',
            f'!1 = !{{i32 {self.opt_level}, !"timber_opt_level", i32 {self.opt_level}}}',
        ]

    # ------------------------------------------------------------------
    # Linear model
    # ------------------------------------------------------------------

    def _emit_linear(self, stage: LinearStage) -> list[str]:
        """Emit timber_infer_single for a linear model."""
        fp = self._fp
        n_features = len(stage.weights) // max(stage.n_classes, 1) if stage.multi_weights else len(stage.weights)
        n_outputs = 1 if stage.n_classes <= 2 else stage.n_classes

        # Emit weight constants
        lines: list[str] = []
        n_w = len(stage.weights)
        w_vals = ", ".join(f"{fp} {self._fp_lit(w)}" for w in stage.weights)
        lines.append(f"@timber_weights = private constant [{n_w} x {fp}] [{w_vals}], align 16")

        if stage.multi_weights:
            b_vals = ", ".join(f"{fp} {self._fp_lit(b)}" for b in (stage.biases or [0.0] * stage.n_classes))
            lines.append(f"@timber_biases  = private constant [{stage.n_classes} x {fp}] [{b_vals}], align 16")
        else:
            lines.append(f"@timber_bias    = private constant {fp} {self._fp_lit(stage.bias)}, align 4")
        lines.append("")

        # Function signature: timber_infer_single(float* inputs, float* outputs) -> i32
        lines += [
            f"; Linear inference: {n_features} features → {n_outputs} outputs",
            f"define i32 @timber_infer_single({fp}* nocapture readonly %inputs, {fp}* nocapture %outputs) local_unnamed_addr #0 {{",
            "entry:",
        ]

        self._reg = 0
        if stage.multi_weights:
            n_cls = stage.n_classes
            for c in range(n_cls):
                # Compute dot product for class c
                acc = self._r()
                lines.append(f"  %{acc} = load {fp}, {fp}* getelementptr inbounds ([{n_cls} x {fp}], [{n_cls} x {fp}]* @timber_biases, i64 0, i64 {c}), align 4")
                dot = acc
                for f_idx in range(n_features):
                    xi = self._r()
                    wi = self._r()
                    prod = self._r()
                    new_dot = self._r()
                    w_offset = c * n_features + f_idx
                    lines.append(f"  %{xi} = load {fp}, {fp}* getelementptr inbounds ([{n_w} x {fp}], [{n_w} x {fp}]* @timber_weights, i64 0, i64 {w_offset}), align 4")
                    lines.append(f"  %{wi} = load {fp}, {fp}* (i64 0, i64 {f_idx})")
                    lines.append(f"  %{wi} = load {fp}, {fp}* getelementptr ({fp}, {fp}* %inputs, i64 {f_idx})")
                    lines.append(f"  %{prod} = fmul {fp} %{xi}, %{wi}")
                    lines.append(f"  %{new_dot} = fadd {fp} %{dot}, %{prod}")
                    dot = new_dot
                # Store result
                lines.append(f"  %out_ptr_{c} = getelementptr inbounds {fp}, {fp}* %outputs, i64 {c}")
                lines.append(f"  store {fp} %{dot}, {fp}* %out_ptr_{c}, align 4")
        else:
            # Binary / regression
            bias_r = self._r()
            lines.append(f"  %{bias_r} = load {fp}, {fp}* @timber_bias, align 4")
            dot = bias_r
            for f_idx in range(n_features):
                xi = self._r()
                wi = self._r()
                prod = self._r()
                new_dot = self._r()
                lines.append(f"  %{xi} = load {fp}, {fp}* getelementptr inbounds ([{n_w} x {fp}], [{n_w} x {fp}]* @timber_weights, i64 0, i64 {f_idx}), align 4")
                lines.append(f"  %{wi} = load {fp}, {fp}* getelementptr ({fp}, {fp}* %inputs, i64 {f_idx})")
                lines.append(f"  %{prod} = fmul {fp} %{xi}, %{wi}")
                lines.append(f"  %{new_dot} = fadd {fp} %{dot}, %{prod}")
                dot = new_dot
            if stage.activation in ("sigmoid", "logistic"):
                neg = self._r()
                exp_r = self._r()
                one_plus = self._r()
                result = self._r()
                lines.append(f"  %{neg} = fneg {fp} %{dot}")
                lines.append(f"  %{exp_r} = call {fp} @llvm.exp.{fp}({fp} %{neg})")
                lines.append(f"  %{one_plus} = fadd {fp} %{exp_r}, {self._fp_lit(1.0)}")
                lines.append(f"  %{result} = fdiv {fp} {self._fp_lit(1.0)}, %{one_plus}")
                lines.append(f"  store {fp} %{result}, {fp}* %outputs, align 4")
            else:
                lines.append(f"  store {fp} %{dot}, {fp}* %outputs, align 4")

        lines += ["  ret i32 0", "}", ""]

        if stage.activation in ("sigmoid", "logistic") and not stage.multi_weights:
            lines.append(f"declare {fp} @llvm.exp.{fp}({fp}) nounwind readnone")
        lines.append('attributes #0 = { nounwind "target-features"="+avx2" }')
        return lines

    # ------------------------------------------------------------------
    # SVM model
    # ------------------------------------------------------------------

    def _emit_svm(self, stage: SVMStage) -> list[str]:
        """Emit kernel function + timber_infer_single for an SVM model."""
        fp = self._fp
        n_sv = stage.n_sv
        n_features = stage.n_features

        lines: list[str] = []

        # Support vector data
        sv_flat = [v for sv in stage.support_vectors for v in sv]
        n_sv_vals = len(sv_flat)
        sv_vals = ", ".join(f"{fp} {self._fp_lit(v)}" for v in sv_flat)
        lines.append(f"@timber_sv       = private constant [{n_sv_vals} x {fp}] [{sv_vals}], align 16")

        dc_vals = ", ".join(f"{fp} {self._fp_lit(v)}" for v in stage.dual_coef)
        lines.append(f"@timber_dual_coef= private constant [{len(stage.dual_coef)} x {fp}] [{dc_vals}], align 16")

        rho_val = stage.rho[0] if stage.rho else 0.0
        lines.append(f"@timber_rho      = private constant {fp} {self._fp_lit(rho_val)}, align 4")
        lines.append(f"@timber_gamma    = private constant {fp} {self._fp_lit(stage.gamma)}, align 4")
        lines.append("")

        # Inference function — unrolled RBF kernel for small n_sv
        lines += [
            f"; SVM ({stage.kernel_type}) inference: {n_features} features, {n_sv} support vectors",
            f"define i32 @timber_infer_single({fp}* nocapture readonly %inputs, {fp}* nocapture %outputs) local_unnamed_addr #0 {{",
            "entry:",
        ]
        self._reg = 0

        # Load rho
        rho_r = self._r()
        lines.append(f"  %{rho_r} = load {fp}, {fp}* @timber_rho, align 4")
        gamma_r = self._r()
        lines.append(f"  %{gamma_r} = load {fp}, {fp}* @timber_gamma, align 4")

        decision = rho_r
        for sv_i in range(n_sv):
            # Compute kernel(inputs, sv[sv_i])
            if stage.kernel_type.lower() == "rbf":
                dist = self._r()
                lines.append(f"  %{dist} = {fp} 0.0{self._fp_suffix}")
                for f_idx in range(n_features):
                    xi = self._r()
                    si = self._r()
                    diff = self._r()
                    dsq = self._r()
                    new_dist = self._r()
                    xi_ptr = sv_i * n_features + f_idx
                    lines.append(f"  %{xi} = load {fp}, {fp}* getelementptr ({fp}, {fp}* %inputs, i64 {f_idx})")
                    lines.append(f"  %{si} = load {fp}, {fp}* getelementptr inbounds ([{n_sv_vals} x {fp}], [{n_sv_vals} x {fp}]* @timber_sv, i64 0, i64 {xi_ptr}), align 4")
                    lines.append(f"  %{diff} = fsub {fp} %{xi}, %{si}")
                    lines.append(f"  %{dsq} = fmul {fp} %{diff}, %{diff}")
                    lines.append(f"  %{new_dist} = fadd {fp} %{dist}, %{dsq}")
                    dist = new_dist
                neg_g_dist = self._r()
                kval = self._r()
                lines.append(f"  %{neg_g_dist} = fmul {fp} %{gamma_r}, %{dist}")
                lines.append(f"  %{neg_g_dist}_neg = fneg {fp} %{neg_g_dist}")
                lines.append(f"  %{kval} = call {fp} @llvm.exp.{fp}({fp} %{neg_g_dist}_neg)")
            else:
                # linear kernel
                dot = self._r()
                lines.append(f"  %{dot} = {fp} 0.0{self._fp_suffix}")
                for f_idx in range(n_features):
                    xi = self._r()
                    si = self._r()
                    prod = self._r()
                    new_dot = self._r()
                    xi_ptr = sv_i * n_features + f_idx
                    lines.append(f"  %{xi} = load {fp}, {fp}* getelementptr ({fp}, {fp}* %inputs, i64 {f_idx})")
                    lines.append(f"  %{si} = load {fp}, {fp}* getelementptr inbounds ([{n_sv_vals} x {fp}], [{n_sv_vals} x {fp}]* @timber_sv, i64 0, i64 {xi_ptr}), align 4")
                    lines.append(f"  %{prod} = fmul {fp} %{xi}, %{si}")
                    lines.append(f"  %{new_dot} = fadd {fp} %{dot}, %{prod}")
                    dot = new_dot
                kval = dot

            # Multiply by dual_coef[sv_i]
            dc_r = self._r()
            contrib = self._r()
            new_decision = self._r()
            lines.append(f"  %{dc_r} = load {fp}, {fp}* getelementptr inbounds ([{len(stage.dual_coef)} x {fp}], [{len(stage.dual_coef)} x {fp}]* @timber_dual_coef, i64 0, i64 {sv_i}), align 4")
            lines.append(f"  %{contrib} = fmul {fp} %{dc_r}, %{kval}")
            lines.append(f"  %{new_decision} = fadd {fp} %{decision}, %{contrib}")
            decision = new_decision

        if stage.post_transform == "logistic":
            neg_d = self._r()
            exp_r = self._r()
            one_plus = self._r()
            result = self._r()
            lines.append(f"  %{neg_d} = fneg {fp} %{decision}")
            lines.append(f"  %{exp_r} = call {fp} @llvm.exp.{fp}({fp} %{neg_d})")
            lines.append(f"  %{one_plus} = fadd {fp} %{exp_r}, {self._fp_lit(1.0)}")
            lines.append(f"  %{result} = fdiv {fp} {self._fp_lit(1.0)}, %{one_plus}")
            lines.append(f"  store {fp} %{result}, {fp}* %outputs, align 4")
        else:
            lines.append(f"  store {fp} %{decision}, {fp}* %outputs, align 4")

        lines += ["  ret i32 0", "}", ""]
        lines.append(f"declare {fp} @llvm.exp.{fp}({fp}) nounwind readnone")
        lines.append('attributes #0 = { nounwind }')
        return lines

    # ------------------------------------------------------------------
    # Tree ensemble
    # ------------------------------------------------------------------

    def _emit_tree_ensemble(self, stage: TreeEnsembleStage) -> list[str]:
        """Emit tree traversal functions in LLVM IR.

        Each tree becomes a function ``@traverse_tree_N`` with explicit
        basic blocks for every internal node.  The top-level
        ``@timber_infer_single`` sums the tree outputs and applies the
        appropriate post-processing.
        """
        fp = self._fp
        lines: list[str] = []

        # Emit per-tree traversal functions
        for tree in stage.trees:
            lines += self._emit_tree_fn(tree, stage, fp)

        n_trees = len(stage.trees)
        n_classes = max(stage.n_classes, 1)
        n_outputs = 1 if n_classes <= 2 else n_classes
        n_features = stage.n_features

        # Top-level inference function
        lines += [
            f"; Top-level inference: {n_trees} trees, {n_features} features",
            f"define i32 @timber_infer_single({fp}* nocapture readonly %inputs, {fp}* nocapture %outputs) local_unnamed_addr #0 {{",
            "entry:",
        ]
        self._reg = 0

        # Accumulate tree predictions
        if n_classes <= 2:
            acc = self._r()
            base = stage.per_class_base_scores[0] if stage.per_class_base_scores else stage.base_score
            lines.append(f"  %{acc} = {fp} {self._fp_lit(base)}")
            for ti, tree in enumerate(stage.trees):
                r = self._r()
                new_acc = self._r()
                lines.append(f"  %{r} = call {fp} @traverse_tree_{ti}({fp}* %inputs)")
                lines.append(f"  %{new_acc} = fadd {fp} %{acc}, %{r}")
                acc = new_acc
            if stage.objective == Objective.BINARY_CLASSIFICATION:
                neg_r = self._r()
                exp_r = self._r()
                denom = self._r()
                result = self._r()
                lines.append(f"  %{neg_r} = fneg {fp} %{acc}")
                lines.append(f"  %{exp_r} = call {fp} @llvm.exp.{fp}({fp} %{neg_r})")
                lines.append(f"  %{denom} = fadd {fp} %{exp_r}, {self._fp_lit(1.0)}")
                lines.append(f"  %{result} = fdiv {fp} {self._fp_lit(1.0)}, %{denom}")
                lines.append(f"  store {fp} %{result}, {fp}* %outputs, align 4")
            else:
                lines.append(f"  store {fp} %{acc}, {fp}* %outputs, align 4")
        else:
            # Multiclass: trees cycle through classes
            for c in range(n_classes):
                base = stage.per_class_base_scores[c] if len(stage.per_class_base_scores) > c else 0.0
                class_trees = [ti for ti, _ in enumerate(stage.trees) if ti % n_classes == c]
                acc = self._r()
                lines.append(f"  %{acc} = {fp} {self._fp_lit(base)}")
                for ti in class_trees:
                    r = self._r()
                    new_acc = self._r()
                    lines.append(f"  %{r} = call {fp} @traverse_tree_{ti}({fp}* %inputs)")
                    lines.append(f"  %{new_acc} = fadd {fp} %{acc}, %{r}")
                    acc = new_acc
                out_ptr = self._r()
                lines.append(f"  %{out_ptr} = getelementptr inbounds {fp}, {fp}* %outputs, i64 {c}")
                lines.append(f"  store {fp} %{acc}, {fp}* %{out_ptr}, align 4")

        lines += ["  ret i32 0", "}", ""]
        lines.append(f"declare {fp} @llvm.exp.{fp}({fp}) nounwind readnone")
        lines.append('attributes #0 = { nounwind "target-features"="" }')
        return lines

    def _emit_tree_fn(
        self, tree: Tree, stage: TreeEnsembleStage, fp: str
    ) -> list[str]:
        """Emit a single tree traversal function using explicit basic blocks."""
        tid = tree.tree_id
        lines = [
            f"; Tree {tid}: {len(tree.nodes)} nodes",
            f"define internal {fp} @traverse_tree_{tid}({fp}* nocapture readonly %inputs) local_unnamed_addr #0 {{",
            "entry:",
            "  br label %node_0",
        ]
        self._reg = 0

        for node in tree.nodes:
            nid = node.node_id
            if node.is_leaf:
                lines += [
                    f"node_{nid}:",
                    f"  ret {fp} {self._fp_lit(node.leaf_value)}",
                ]
            else:
                fidx = node.feature_index
                thr = node.threshold
                feat_r = self._r()
                cmp_r = self._r()
                true_id = node.left_child if node.left_child >= 0 else nid + 1
                false_id = node.right_child if node.right_child >= 0 else nid + 1
                lines += [
                    f"node_{nid}:",
                    f"  %{feat_r} = load {fp}, {fp}* getelementptr ({fp}, {fp}* %inputs, i64 {fidx})",
                    f"  %{cmp_r} = fcmp olt {fp} %{feat_r}, {self._fp_lit(thr)}",
                    f"  br i1 %{cmp_r}, label %node_{true_id}, label %node_{false_id}",
                ]

        lines += ["}", ""]
        return lines

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _r(self) -> str:
        """Allocate a fresh SSA register name."""
        self._reg += 1
        return f"r{self._reg}"

    def _fp_lit(self, v: float) -> str:
        """Format a float as an LLVM IR literal (hex float for precision)."""
        if self.use_float:
            import struct
            b = struct.pack(">f", v)
            u32 = int.from_bytes(b, "big")
            # LLVM uses 64-bit hex float literals even for f32
            u64 = struct.unpack(">d", struct.pack(">d", float(v)))[0]
            hex_bits = struct.pack(">d", float(v)).hex()
            return f"0x{struct.unpack('>Q', struct.pack('>d', float(v)))[0]:016X}"
        else:
            import struct
            return f"0x{struct.unpack('>Q', struct.pack('>d', float(v)))[0]:016X}"
