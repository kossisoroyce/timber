"""Tests for the Timber IR data model and serialization."""

import json
import pytest

from timber.ir.model import (
    TimberIR,
    Schema,
    Field,
    FieldType,
    Metadata,
    TreeEnsembleStage,
    TreeNode,
    Tree,
    ScalerStage,
    LinearStage,
    Objective,
    PrecisionMode,
)


def _make_simple_tree(tree_id: int = 0) -> Tree:
    """Create a simple 3-node tree: root splits on feature 0, two leaves."""
    nodes = [
        TreeNode(node_id=0, feature_index=0, threshold=0.5,
                 left_child=1, right_child=2, is_leaf=False, depth=0),
        TreeNode(node_id=1, is_leaf=True, leaf_value=-0.3, depth=1),
        TreeNode(node_id=2, is_leaf=True, leaf_value=0.7, depth=1),
    ]
    tree = Tree(tree_id=tree_id, nodes=nodes)
    tree.recount()
    return tree


def _make_simple_ir() -> TimberIR:
    """Create a minimal IR with one tree ensemble."""
    tree = _make_simple_tree()
    ensemble = TreeEnsembleStage(
        stage_name="test_ensemble",
        stage_type="tree_ensemble",
        trees=[tree],
        n_features=3,
        n_classes=1,
        objective=Objective.REGRESSION,
        base_score=0.5,
    )
    schema = Schema(
        input_fields=[
            Field(name="f0", dtype=FieldType.FLOAT32, index=0),
            Field(name="f1", dtype=FieldType.FLOAT32, index=1),
            Field(name="f2", dtype=FieldType.FLOAT32, index=2),
        ],
        output_fields=[
            Field(name="output_0", dtype=FieldType.FLOAT32, index=0),
        ],
    )
    metadata = Metadata(
        source_framework="test",
        feature_names=["f0", "f1", "f2"],
    )
    return TimberIR(pipeline=[ensemble], schema=schema, metadata=metadata)


class TestTreeNode:
    def test_create_leaf(self):
        node = TreeNode(node_id=0, is_leaf=True, leaf_value=1.5)
        assert node.is_leaf
        assert node.leaf_value == 1.5
        assert node.feature_index == -1

    def test_create_internal(self):
        node = TreeNode(node_id=0, feature_index=2, threshold=3.14,
                        left_child=1, right_child=2)
        assert not node.is_leaf
        assert node.feature_index == 2
        assert node.threshold == pytest.approx(3.14)

    def test_copy(self):
        node = TreeNode(node_id=0, is_leaf=True, leaf_value=1.0)
        copied = node.copy()
        copied.leaf_value = 2.0
        assert node.leaf_value == 1.0


class TestTree:
    def test_recount(self):
        tree = _make_simple_tree()
        assert tree.n_leaves == 2
        assert tree.n_internal == 1
        assert tree.max_depth == 1

    def test_empty_tree(self):
        tree = Tree(tree_id=0)
        tree.recount()
        assert tree.n_leaves == 0
        assert tree.n_internal == 0


class TestTimberIR:
    def test_summary(self):
        ir = _make_simple_ir()
        summary = ir.summary()
        assert summary["n_trees"] == 1
        assert summary["max_depth"] == 1
        assert summary["n_features"] == 3
        assert summary["n_input_features"] == 3

    def test_get_tree_ensemble(self):
        ir = _make_simple_ir()
        ensemble = ir.get_tree_ensemble()
        assert ensemble is not None
        assert ensemble.n_trees == 1

    def test_serialization_roundtrip(self):
        ir = _make_simple_ir()
        json_str = ir.to_json()
        restored = TimberIR.from_json(json_str)

        assert len(restored.pipeline) == 1
        ensemble = restored.get_tree_ensemble()
        assert ensemble is not None
        assert ensemble.n_trees == 1
        assert ensemble.trees[0].nodes[0].threshold == 0.5
        assert ensemble.trees[0].nodes[1].leaf_value == -0.3
        assert ensemble.trees[0].nodes[2].leaf_value == 0.7

    def test_deep_copy(self):
        ir = _make_simple_ir()
        copied = ir.deep_copy()
        ensemble = copied.get_tree_ensemble()
        ensemble.trees[0].nodes[0].threshold = 999.0
        # Original should be unchanged
        assert ir.get_tree_ensemble().trees[0].nodes[0].threshold == 0.5

    def test_schema_properties(self):
        ir = _make_simple_ir()
        assert ir.schema.n_features == 3
        assert ir.schema.n_outputs == 1


class TestScalerStage:
    def test_create(self):
        scaler = ScalerStage(
            stage_name="std_scaler",
            stage_type="scaler",
            means=[0.0, 1.0, 2.0],
            scales=[1.0, 2.0, 3.0],
            feature_indices=[0, 1, 2],
        )
        assert scaler.stage_type == "scaler"
        assert len(scaler.means) == 3


class TestLinearStage:
    def test_create(self):
        linear = LinearStage(
            stage_name="logistic",
            stage_type="linear",
            weights=[0.1, -0.2, 0.3],
            bias=0.5,
            activation="sigmoid",
        )
        assert linear.stage_type == "linear"
        assert linear.activation == "sigmoid"
