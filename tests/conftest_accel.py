"""Shared fixtures for timber.accel tests — imported via conftest.py."""

import json

import pytest

from timber.ir.model import (
    Field,
    FieldType,
    LinearStage,
    Metadata,
    Objective,
    Schema,
    TimberIR,
    Tree,
    TreeEnsembleStage,
    TreeNode,
)


@pytest.fixture
def simple_schema():
    return Schema(
        input_fields=[Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i) for i in range(4)],
        output_fields=[Field(name="output", dtype=FieldType.FLOAT32, index=0)],
    )


@pytest.fixture
def simple_metadata():
    return Metadata(
        source_framework="test",
        source_framework_version="1.0",
        source_artifact_hash="deadbeef",
        feature_names=["f0", "f1", "f2", "f3"],
        feature_importances=[0.4, 0.3, 0.2, 0.1],
        objective_name="regression",
        training_params={},
        compilation_hints={},
    )


@pytest.fixture
def simple_tree():
    return Tree(
        tree_id=0,
        nodes=[
            TreeNode(node_id=0, feature_index=0, threshold=0.5,
                     left_child=1, right_child=2, is_leaf=False, depth=0),
            TreeNode(node_id=1, is_leaf=True, leaf_value=1.0, depth=1),
            TreeNode(node_id=2, is_leaf=True, leaf_value=-1.0, depth=1),
        ],
        max_depth=1, n_leaves=2, n_internal=1,
    )


@pytest.fixture
def deeper_tree():
    return Tree(
        tree_id=0,
        nodes=[
            TreeNode(node_id=0, feature_index=0, threshold=0.5,
                     left_child=1, right_child=2, is_leaf=False, depth=0),
            TreeNode(node_id=1, feature_index=1, threshold=0.3,
                     left_child=3, right_child=4, is_leaf=False, depth=1),
            TreeNode(node_id=2, feature_index=2, threshold=0.7,
                     left_child=5, right_child=6, is_leaf=False, depth=1),
            TreeNode(node_id=3, is_leaf=True, leaf_value=1.0, depth=2),
            TreeNode(node_id=4, is_leaf=True, leaf_value=0.5, depth=2),
            TreeNode(node_id=5, is_leaf=True, leaf_value=-0.5, depth=2),
            TreeNode(node_id=6, is_leaf=True, leaf_value=-1.0, depth=2),
        ],
        max_depth=2, n_leaves=4, n_internal=3,
    )


@pytest.fixture
def simple_ensemble(simple_tree, simple_schema, simple_metadata):
    stage = TreeEnsembleStage(
        stage_name="trees", stage_type="tree_ensemble",
        trees=[simple_tree], n_features=4, n_classes=1,
        objective=Objective.REGRESSION, base_score=0.0,
    )
    return TimberIR(pipeline=[stage], schema=simple_schema, metadata=simple_metadata)


@pytest.fixture
def multi_tree_ensemble(simple_tree, deeper_tree, simple_schema, simple_metadata):
    tree2 = Tree(tree_id=1, nodes=deeper_tree.nodes,
                 max_depth=deeper_tree.max_depth,
                 n_leaves=deeper_tree.n_leaves,
                 n_internal=deeper_tree.n_internal)
    stage = TreeEnsembleStage(
        stage_name="trees", stage_type="tree_ensemble",
        trees=[simple_tree, tree2], n_features=4, n_classes=1,
        objective=Objective.REGRESSION, base_score=0.0,
    )
    return TimberIR(pipeline=[stage], schema=simple_schema, metadata=simple_metadata)


@pytest.fixture
def linear_ir(simple_schema, simple_metadata):
    stage = LinearStage(
        stage_name="linear", stage_type="linear",
        weights=[0.5, -0.3, 0.8, 0.1], bias=0.2,
        activation="none", n_classes=1,
    )
    return TimberIR(pipeline=[stage], schema=simple_schema, metadata=simple_metadata)


@pytest.fixture
def binary_classification_ensemble(simple_tree, deeper_tree, simple_schema, simple_metadata):
    stage = TreeEnsembleStage(
        stage_name="trees", stage_type="tree_ensemble",
        trees=[simple_tree, Tree(tree_id=1, nodes=deeper_tree.nodes,
               max_depth=deeper_tree.max_depth, n_leaves=deeper_tree.n_leaves,
               n_internal=deeper_tree.n_internal)],
        n_features=4, n_classes=1,
        objective=Objective.BINARY_CLASSIFICATION, base_score=0.5,
    )
    return TimberIR(pipeline=[stage], schema=simple_schema, metadata=simple_metadata)


def _make_xgb_json(n_features=4, n_trees=2) -> str:
    trees = []
    for i in range(n_trees):
        trees.append({
            "tree_param": {"num_nodes": "3", "size_leaf_vector": "0",
                           "num_feature": str(n_features)},
            "id": i, "num_nodes": 3,
            "split_indices": [0, 0, 0], "split_conditions": [0.5, 0.0, 0.0],
            "split_type": [0, 0, 0], "left_children": [1, -1, -1],
            "right_children": [2, -1, -1], "parents": [2147483647, 0, 0],
            "default_left": [1, 0, 0], "categories": [],
            "categories_nodes": [], "categories_segments": [],
            "categories_sizes": [], "base_weights": [0.0, 0.3, -0.3],
            "loss_changes": [0.0, 0.0, 0.0], "sum_hessian": [0.0, 0.0, 0.0],
        })
    return json.dumps({
        "learner": {
            "learner_model_param": {
                "num_feature": str(n_features), "num_class": "0", "base_score": "5E-1",
            },
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "gbtree_model_param": {"num_trees": str(n_trees), "size_leaf_vector": "0"},
                    "trees": trees, "tree_info": [0] * n_trees,
                },
            },
            "objective": {"name": "reg:squarederror",
                          "reg_loss_param": {"scale_pos_weight": "1"}},
        },
        "version": [2, 0, 3],
    })


@pytest.fixture
def xgb_model_path(tmp_path):
    path = tmp_path / "test_model.json"
    path.write_text(_make_xgb_json())
    return str(path)
