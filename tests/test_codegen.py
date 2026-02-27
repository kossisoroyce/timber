"""Tests for the C99 code emitter."""

import pytest
from pathlib import Path

from timber.ir.model import (
    TimberIR,
    Schema,
    Field,
    FieldType,
    Metadata,
    TreeEnsembleStage,
    TreeNode,
    Tree,
    Objective,
)
from timber.codegen.c99 import C99Emitter, TargetSpec, C99Output


def _make_simple_ir(n_trees=2, objective=Objective.REGRESSION) -> TimberIR:
    """Create a minimal IR for codegen testing."""
    trees = []
    for tid in range(n_trees):
        nodes = [
            TreeNode(node_id=0, feature_index=0, threshold=0.5,
                     left_child=1, right_child=2, is_leaf=False, depth=0),
            TreeNode(node_id=1, is_leaf=True, leaf_value=-0.3 * (tid + 1), depth=1),
            TreeNode(node_id=2, is_leaf=True, leaf_value=0.7 * (tid + 1), depth=1),
        ]
        tree = Tree(tree_id=tid, nodes=nodes, max_depth=1, n_leaves=2, n_internal=1)
        trees.append(tree)

    n_classes = 1
    if objective == Objective.BINARY_CLASSIFICATION:
        n_classes = 2

    ensemble = TreeEnsembleStage(
        stage_name="test_ensemble",
        stage_type="tree_ensemble",
        trees=trees,
        n_features=3,
        n_classes=n_classes,
        objective=objective,
        base_score=0.5,
    )
    schema = Schema(
        input_fields=[Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i) for i in range(3)],
        output_fields=[Field(name="output_0", dtype=FieldType.FLOAT32, index=0)],
    )
    return TimberIR(pipeline=[ensemble], schema=schema, metadata=Metadata())


class TestC99Emitter:
    def test_emit_regression(self):
        ir = _make_simple_ir(n_trees=2, objective=Objective.REGRESSION)
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert isinstance(output, C99Output)
        assert "timber_infer" in output.model_h
        assert "timber_infer" in output.model_c
        assert "TIMBER_N_FEATURES" in output.model_h
        assert "#define TIMBER_N_FEATURES 3" in output.model_h
        assert "#define TIMBER_N_TREES    2" in output.model_h

    def test_emit_binary_classification(self):
        ir = _make_simple_ir(n_trees=2, objective=Objective.BINARY_CLASSIFICATION)
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert "exp(-sum)" in output.model_c

    def test_emit_data(self):
        ir = _make_simple_ir(n_trees=1)
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert "tree_0_features" in output.model_data_c
        assert "tree_0_thresholds" in output.model_data_c
        assert "tree_0_leaves" in output.model_data_c
        assert "TIMBER_BASE_SCORE" in output.model_data_c

    def test_emit_cmake(self):
        ir = _make_simple_ir()
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert "cmake_minimum_required" in output.cmakelists
        assert "timber_model" in output.cmakelists

    def test_emit_makefile(self):
        ir = _make_simple_ir()
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert "libtimber_model.so" in output.makefile
        assert "-std=c99" in output.makefile

    def test_emit_avx512_flags(self):
        ir = _make_simple_ir()
        target = TargetSpec(features=["avx512f", "avx512bw"])
        emitter = C99Emitter(target=target)
        output = emitter.emit(ir)

        assert "avx512" in output.makefile
        assert "avx512" in output.cmakelists

    def test_write_output(self, tmp_path):
        ir = _make_simple_ir()
        emitter = C99Emitter()
        output = emitter.emit(ir)

        files = output.write(tmp_path / "dist")
        assert len(files) == 5
        for f in files:
            assert Path(f).exists()

    def test_no_ensemble_raises(self):
        ir = TimberIR()
        emitter = C99Emitter()
        with pytest.raises(ValueError, match="No tree ensemble"):
            emitter.emit(ir)

    def test_header_c_abi(self):
        ir = _make_simple_ir()
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert '#ifdef __cplusplus' in output.model_h
        assert 'extern "C"' in output.model_h
        assert '#endif' in output.model_h

    def test_traverse_function(self):
        ir = _make_simple_ir()
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert "traverse_tree" in output.model_c
        assert "NaN check" in output.model_c
