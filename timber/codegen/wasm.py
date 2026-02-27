"""WebAssembly emitter — generates a WAT (WebAssembly Text) module from Timber IR.

The emitter produces a .wat file that can be compiled to .wasm using wat2wasm,
or consumed directly by runtimes like wasmtime/wasmer. The generated module
exports `timber_infer_single` and operates on a linear memory buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from timber.ir.model import (
    Objective,
    TimberIR,
    TreeEnsembleStage,
)


@dataclass
class WasmOutput:
    """Output container for the WASM emitter."""
    wat: str
    js_bindings: str

    def write(self, output_dir: str | Path) -> list[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = []

        wat_path = output_dir / "model.wat"
        wat_path.write_text(self.wat, encoding="utf-8")
        files.append(str(wat_path))

        js_path = output_dir / "timber_model.js"
        js_path.write_text(self.js_bindings, encoding="utf-8")
        files.append(str(js_path))

        return files


class WasmEmitter:
    """Emits WebAssembly Text format (WAT) from optimized Timber IR."""

    def emit(self, ir: TimberIR) -> WasmOutput:
        ensemble = ir.get_tree_ensemble()
        if ensemble is None:
            raise ValueError("No tree ensemble found in IR pipeline")

        wat = self._emit_wat(ir, ensemble)
        js = self._emit_js_bindings(ir, ensemble)
        return WasmOutput(wat=wat, js_bindings=js)

    def _emit_wat(self, ir: TimberIR, ensemble: TreeEnsembleStage) -> str:
        n_features = ensemble.n_features
        n_outputs = 1 if ensemble.n_classes <= 2 else ensemble.n_classes
        lines = [
            '(module',
            '  ;; Timber compiled model — WebAssembly',
            f'  ;; Trees: {ensemble.n_trees}, Features: {n_features}, Outputs: {n_outputs}',
            '',
            '  ;; Linear memory: inputs at offset 0, outputs after inputs',
            f'  (memory (export "memory") 1)',
            '',
        ]

        # Emit tree data as data segments
        data_offset = (n_features + n_outputs) * 4 + 64  # after I/O buffers
        tree_offsets = []

        for tree in ensemble.trees:
            n_nodes = len(tree.nodes)
            tree_offsets.append({
                'n_nodes': n_nodes,
                'features_off': data_offset,
                'thresholds_off': data_offset + n_nodes * 4,
                'left_off': data_offset + n_nodes * 8,
                'right_off': data_offset + n_nodes * 12,
                'leaves_off': data_offset + n_nodes * 16,
                'is_leaf_off': data_offset + n_nodes * 20,
                'default_left_off': data_offset + n_nodes * 21,
            })

            # Feature indices (i32)
            feat_bytes = b''.join(int(n.feature_index).to_bytes(4, 'little', signed=True) for n in tree.nodes)
            lines.append(f'  (data (i32.const {data_offset}) "{self._escape_bytes(feat_bytes)}")')
            data_offset += n_nodes * 4

            # Thresholds (f32)
            import struct
            thresh_bytes = b''.join(struct.pack('<f', n.threshold) for n in tree.nodes)
            lines.append(f'  (data (i32.const {data_offset}) "{self._escape_bytes(thresh_bytes)}")')
            data_offset += n_nodes * 4

            # Left children (i32)
            left_bytes = b''.join(int(n.left_child).to_bytes(4, 'little', signed=True) for n in tree.nodes)
            lines.append(f'  (data (i32.const {data_offset}) "{self._escape_bytes(left_bytes)}")')
            data_offset += n_nodes * 4

            # Right children (i32)
            right_bytes = b''.join(int(n.right_child).to_bytes(4, 'little', signed=True) for n in tree.nodes)
            lines.append(f'  (data (i32.const {data_offset}) "{self._escape_bytes(right_bytes)}")')
            data_offset += n_nodes * 4

            # Leaf values (f32)
            leaf_bytes = b''.join(struct.pack('<f', n.leaf_value) for n in tree.nodes)
            lines.append(f'  (data (i32.const {data_offset}) "{self._escape_bytes(leaf_bytes)}")')
            data_offset += n_nodes * 4

            # Is-leaf (i8)
            is_leaf_bytes = bytes(1 if n.is_leaf else 0 for n in tree.nodes)
            lines.append(f'  (data (i32.const {data_offset}) "{self._escape_bytes(is_leaf_bytes)}")')
            data_offset += n_nodes

            # Default-left (i8)
            dl_bytes = bytes(1 if n.default_left else 0 for n in tree.nodes)
            lines.append(f'  (data (i32.const {data_offset}) "{self._escape_bytes(dl_bytes)}")')
            data_offset += n_nodes

        lines.append('')

        # Store base_score
        import struct
        bs_off = data_offset
        bs_bytes = struct.pack('<f', ensemble.base_score)
        lines.append(f'  (data (i32.const {bs_off}) "{self._escape_bytes(bs_bytes)}")')
        data_offset += 4
        lines.append('')

        # Traverse tree function
        lines.extend([
            '  ;; traverse_tree(input_ptr, feat_off, thresh_off, left_off, right_off, leaf_off, is_leaf_off, dl_off, n_nodes) -> f32',
            '  (func $traverse_tree (param $inp i32) (param $feat i32) (param $thresh i32) (param $left i32) (param $right i32) (param $leaf i32) (param $is_leaf i32) (param $dl i32) (param $nn i32) (result f32)',
            '    (local $node i32)',
            '    (local $iter i32)',
            '    (local $fi i32)',
            '    (local $val f32)',
            '    (local.set $node (i32.const 0))',
            f'    (local.set $iter (i32.const {ensemble.max_depth + 2}))',
            '    (block $done',
            '      (loop $loop',
            '        ;; bounds check',
            '        (br_if $done (i32.lt_s (local.get $iter) (i32.const 1)))',
            '        (local.set $iter (i32.sub (local.get $iter) (i32.const 1)))',
            '        (br_if $done (i32.lt_s (local.get $node) (i32.const 0)))',
            '        (br_if $done (i32.ge_s (local.get $node) (local.get $nn)))',
            '        ;; is_leaf check',
            '        (if (i32.load8_u (i32.add (local.get $is_leaf) (local.get $node)))',
            '          (then',
            '            (return (f32.load (i32.add (local.get $leaf) (i32.mul (local.get $node) (i32.const 4)))))',
            '          )',
            '        )',
            '        ;; get feature index and value',
            '        (local.set $fi (i32.load (i32.add (local.get $feat) (i32.mul (local.get $node) (i32.const 4)))))',
            '        (local.set $val (f32.load (i32.add (local.get $inp) (i32.mul (local.get $fi) (i32.const 4)))))',
            '        ;; NaN check (val != val)',
            '        (if (i32.eqz (f32.eq (local.get $val) (local.get $val)))',
            '          (then',
            '            (if (i32.load8_u (i32.add (local.get $dl) (local.get $node)))',
            '              (then (local.set $node (i32.load (i32.add (local.get $left) (i32.mul (local.get $node) (i32.const 4))))))',
            '              (else (local.set $node (i32.load (i32.add (local.get $right) (i32.mul (local.get $node) (i32.const 4))))))',
            '            )',
            '          )',
            '          (else',
            '            (if (f32.lt (local.get $val) (f32.load (i32.add (local.get $thresh) (i32.mul (local.get $node) (i32.const 4)))))',
            '              (then (local.set $node (i32.load (i32.add (local.get $left) (i32.mul (local.get $node) (i32.const 4))))))',
            '              (else (local.set $node (i32.load (i32.add (local.get $right) (i32.mul (local.get $node) (i32.const 4))))))',
            '            )',
            '          )',
            '        )',
            '        (br $loop)',
            '      )',
            '    )',
            '    (f32.const 0)',
            '  )',
            '',
        ])

        # Infer single function
        input_ptr = 0
        output_ptr = n_features * 4
        lines.extend([
            f'  ;; timber_infer_single: reads {n_features} floats from offset 0, writes {n_outputs} floats to offset {output_ptr}',
            '  (func $timber_infer_single (export "timber_infer_single") (result i32)',
            '    (local $sum f64)',
            f'    (local.set $sum (f64.promote_f32 (f32.load (i32.const {bs_off}))))',
        ])

        for i, tree in enumerate(ensemble.trees):
            off = tree_offsets[i]
            lines.append(
                f'    (local.set $sum (f64.add (local.get $sum) (f64.promote_f32 (call $traverse_tree '
                f'(i32.const {input_ptr}) (i32.const {off["features_off"]}) (i32.const {off["thresholds_off"]}) '
                f'(i32.const {off["left_off"]}) (i32.const {off["right_off"]}) (i32.const {off["leaves_off"]}) '
                f'(i32.const {off["is_leaf_off"]}) (i32.const {off["default_left_off"]}) (i32.const {off["n_nodes"]})))))'
            )

        # Apply activation
        if ensemble.objective in (Objective.BINARY_CLASSIFICATION, Objective.REGRESSION_LOGISTIC):
            # sigmoid: 1 / (1 + exp(-sum))
            # WASM doesn't have exp, so we use an export helper or polynomial approx
            lines.extend([
                '    ;; sigmoid approximation via f64 arithmetic',
                '    (local.set $sum (f64.div (f64.const 1.0) (f64.add (f64.const 1.0) (call $exp_neg (local.get $sum)))))',
            ])

        lines.extend([
            f'    (f32.store (i32.const {output_ptr}) (f32.demote_f64 (local.get $sum)))',
            '    (i32.const 0)',
            '  )',
            '',
        ])

        # exp(-x) helper using Taylor series (good enough for sigmoid range)
        if ensemble.objective in (Objective.BINARY_CLASSIFICATION, Objective.REGRESSION_LOGISTIC):
            lines.extend([
                '  ;; exp(-x) via exp(x) = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720',
                '  ;; Clamped to [-20, 20] for numerical stability',
                '  (func $exp_neg (param $x f64) (result f64)',
                '    (local $nx f64)',
                '    (local $t f64)',
                '    (local $result f64)',
                '    ;; nx = -x, clamped',
                '    (local.set $nx (f64.neg (local.get $x)))',
                '    (if (f64.gt (local.get $nx) (f64.const 20.0)) (then (return (f64.const 485165195.4))))',
                '    (if (f64.lt (local.get $nx) (f64.const -20.0)) (then (return (f64.const 0.0))))',
                '    ;; Horner form: 1 + nx*(1 + nx/2*(1 + nx/3*(1 + nx/4*(1 + nx/5*(1 + nx/6)))))',
                '    (local.set $result (f64.add (f64.const 1.0) (f64.mul (local.get $nx) (f64.div (f64.const 1.0) (f64.const 6.0)))))',
                '    (local.set $result (f64.add (f64.const 1.0) (f64.mul (local.get $nx) (f64.mul (f64.div (f64.const 1.0) (f64.const 5.0)) (local.get $result)))))',
                '    (local.set $result (f64.add (f64.const 1.0) (f64.mul (local.get $nx) (f64.mul (f64.div (f64.const 1.0) (f64.const 4.0)) (local.get $result)))))',
                '    (local.set $result (f64.add (f64.const 1.0) (f64.mul (local.get $nx) (f64.mul (f64.div (f64.const 1.0) (f64.const 3.0)) (local.get $result)))))',
                '    (local.set $result (f64.add (f64.const 1.0) (f64.mul (local.get $nx) (f64.mul (f64.div (f64.const 1.0) (f64.const 2.0)) (local.get $result)))))',
                '    (local.set $result (f64.add (f64.const 1.0) (f64.mul (local.get $nx) (local.get $result))))',
                '    (local.get $result)',
                '  )',
                '',
            ])

        lines.append(')')
        return '\n'.join(lines) + '\n'

    def _emit_js_bindings(self, ir: TimberIR, ensemble: TreeEnsembleStage) -> str:
        n_features = ensemble.n_features
        n_outputs = 1 if ensemble.n_classes <= 2 else ensemble.n_classes
        output_ptr = n_features * 4

        return f"""// timber_model.js — JavaScript bindings for Timber WASM model
// Generated by Timber v0.1

export async function loadTimberModel(wasmPath) {{
  const response = await fetch(wasmPath);
  const bytes = await response.arrayBuffer();
  const {{ instance }} = await WebAssembly.instantiate(bytes);
  const memory = instance.exports.memory;

  return {{
    predict(features) {{
      const input = new Float32Array(memory.buffer, 0, {n_features});
      for (let i = 0; i < {n_features}; i++) input[i] = features[i];
      const rc = instance.exports.timber_infer_single();
      const output = new Float32Array(memory.buffer, {output_ptr}, {n_outputs});
      return Array.from(output);
    }},

    N_FEATURES: {n_features},
    N_OUTPUTS: {n_outputs},
    N_TREES: {ensemble.n_trees},
  }};
}}
"""

    @staticmethod
    def _escape_bytes(data: bytes) -> str:
        return ''.join(f'\\{b:02x}' for b in data)
