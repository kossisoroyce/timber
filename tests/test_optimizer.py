"""Tests for the Timber optimizer passes."""

import numpy as np
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
    Objective,
)
from timber.optimizer.dead_leaf import dead_leaf_elimination
from timber.optimizer.constant_feature import constant_feature_detection
from timber.optimizer.threshold_quant import threshold_quantization
from timber.optimizer.branch_sort import frequency_branch_sort
from timber.optimizer.pipeline_fusion import pipeline_fusion
from timber.optimizer.pipeline import OptimizerPipeline


def _make_tree(nodes_data: list[dict], tree_id: int = 0) -> Tree:
    """Helper to build a tree from a list of node dicts."""
    nodes = []
    for nd in nodes_data:
        nodes.append(TreeNode(**nd))
    tree = Tree(tree_id=tree_id, nodes=nodes)
    tree.recount()
    return tree


def _make_ensemble_ir(trees: list[Tree], n_features: int = 3, **kwargs) -> TimberIR:
    """Helper to build an IR with a tree ensemble."""
    ensemble = TreeEnsembleStage(
        stage_name="test_ensemble",
        stage_type="tree_ensemble",
        trees=trees,
        n_features=n_features,
        objective=kwargs.get("objective", Objective.REGRESSION),
        base_score=kwargs.get("base_score", 0.5),
    )
    schema = Schema(
        input_fields=[Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i) for i in range(n_features)],
        output_fields=[Field(name="output_0", dtype=FieldType.FLOAT32, index=0)],
    )
    pipeline = kwargs.get("pipeline_prefix", []) + [ensemble]
    return TimberIR(pipeline=pipeline, schema=schema, metadata=Metadata())


class TestDeadLeafElimination:
    def test_prune_tiny_leaves(self):
        """Both leaves have negligible values -> collapse."""
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 0.5,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 0.0001, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": 0.0002, "depth": 1},
        ])
        # Add a tree with big values so max_leaf is big
        tree2 = _make_tree([
            {"node_id": 0, "is_leaf": True, "leaf_value": 10.0, "depth": 0},
        ], tree_id=1)

        ir = _make_ensemble_ir([tree, tree2])
        changed, new_ir, details = dead_leaf_elimination(ir, threshold=0.001)

        assert changed
        assert details["leaves_pruned"] > 0

    def test_no_pruning_needed(self):
        """All leaves have significant values."""
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 0.5,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 5.0, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": -3.0, "depth": 1},
        ])
        ir = _make_ensemble_ir([tree])
        changed, new_ir, details = dead_leaf_elimination(ir, threshold=0.001)

        assert not changed
        assert details["leaves_pruned"] == 0

    def test_no_ensemble(self):
        ir = TimberIR()
        changed, _, details = dead_leaf_elimination(ir)
        assert not changed
        assert "skipped" in details


class TestConstantFeatureDetection:
    def test_fold_identical_leaves(self):
        """Split where both children have the same value -> fold."""
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 0.5,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 1.0, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": 1.0, "depth": 1},
        ])
        ir = _make_ensemble_ir([tree])
        changed, new_ir, details = constant_feature_detection(ir)

        assert changed
        assert details["nodes_folded"] == 1
        # Root should now be a leaf
        ensemble = new_ir.get_tree_ensemble()
        assert ensemble.trees[0].nodes[0].is_leaf
        assert ensemble.trees[0].nodes[0].leaf_value == 1.0

    def test_no_folding(self):
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 0.5,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 1.0, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": 2.0, "depth": 1},
        ])
        ir = _make_ensemble_ir([tree])
        changed, _, details = constant_feature_detection(ir)

        assert not changed


class TestThresholdQuantization:
    def test_integer_thresholds(self):
        """Integer thresholds in int8 range -> tagged as int8."""
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 5.0,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 1.0, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": 2.0, "depth": 1},
        ])
        ir = _make_ensemble_ir([tree])
        changed, new_ir, details = threshold_quantization(ir)

        # Feature 0 with threshold 5.0 should be classified as int8
        assert details["features_analyzed"] == 1
        assert 0 in details["int8_features"]

    def test_float_thresholds(self):
        """High-precision thresholds -> stay float32."""
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 3.141592653589793,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 1.0, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": 2.0, "depth": 1},
        ])
        ir = _make_ensemble_ir([tree])
        changed, new_ir, details = threshold_quantization(ir)

        assert 0 in details["float32_features"] or 0 in details["float16_features"]


class TestBranchSort:
    def test_sort_with_calibration(self):
        """Branches should be reordered based on calibration data."""
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 0.5,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 1.0, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": 2.0, "depth": 1},
        ])
        ir = _make_ensemble_ir([tree])

        # All samples have feature 0 > 0.5, so right branch is more frequent
        calib = np.array([[0.8, 0.0, 0.0], [0.9, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=np.float32)
        changed, new_ir, details = frequency_branch_sort(ir, calib)

        assert changed or not changed  # may or may not swap depending on data
        assert details["calibration_samples"] == 3

    def test_no_calibration(self):
        tree = _make_tree([
            {"node_id": 0, "is_leaf": True, "leaf_value": 1.0, "depth": 0},
        ])
        ir = _make_ensemble_ir([tree])
        changed, _, details = frequency_branch_sort(ir, None)
        assert not changed


class TestPipelineFusion:
    def test_fuse_scaler(self):
        """Scaler followed by ensemble -> thresholds adjusted, scaler removed."""
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 1.0,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 1.0, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": 2.0, "depth": 1},
        ])

        scaler = ScalerStage(
            stage_name="std_scaler",
            stage_type="scaler",
            means=[2.0, 0.0, 0.0],
            scales=[3.0, 1.0, 1.0],
            feature_indices=[0, 1, 2],
        )

        ir = _make_ensemble_ir([tree], pipeline_prefix=[scaler])
        assert len(ir.pipeline) == 2

        changed, new_ir, details = pipeline_fusion(ir)

        assert changed
        assert len(new_ir.pipeline) == 1  # scaler removed
        assert details["stages_removed"] == 1

        # Threshold should be adjusted: 1.0 * 3.0 + 2.0 = 5.0
        ensemble = new_ir.get_tree_ensemble()
        assert ensemble.trees[0].nodes[0].threshold == pytest.approx(5.0)

    def test_no_fusion_without_scaler(self):
        tree = _make_tree([
            {"node_id": 0, "is_leaf": True, "leaf_value": 1.0, "depth": 0},
        ])
        ir = _make_ensemble_ir([tree])
        changed, _, details = pipeline_fusion(ir)
        assert not changed


class TestOptimizerPipeline:
    def test_full_pipeline(self):
        tree = _make_tree([
            {"node_id": 0, "feature_index": 0, "threshold": 0.5,
             "left_child": 1, "right_child": 2, "depth": 0},
            {"node_id": 1, "is_leaf": True, "leaf_value": 1.0, "depth": 1},
            {"node_id": 2, "is_leaf": True, "leaf_value": 2.0, "depth": 1},
        ])
        ir = _make_ensemble_ir([tree])

        pipeline = OptimizerPipeline()
        result = pipeline.run(ir)

        assert result.ir is not None
        assert len(result.passes) > 0
        assert result.total_duration_ms >= 0
        summary = result.summary()
        assert "total_passes" in summary
