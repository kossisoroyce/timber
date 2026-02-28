"""Tests for the XGBoost front-end parser."""

import json
import tempfile
import pytest
from pathlib import Path

from timber.frontends.xgboost_parser import parse_xgboost_json, _parse_xgboost_dict
from timber.ir.model import Objective


def _make_xgboost_json(n_trees=2, max_depth=2, n_features=3, objective="reg:squarederror"):
    """Create a synthetic XGBoost JSON model dict."""
    trees = []
    for tid in range(n_trees):
        # Simple depth-2 tree: root -> 2 internal -> 4 leaves
        n_nodes = 2 ** (max_depth + 1) - 1
        split_indices = []
        split_conditions = []
        left_children = []
        right_children = []
        default_left = []

        for nid in range(n_nodes):
            depth = 0
            tmp = nid + 1
            while tmp > 1:
                tmp //= 2
                depth += 1

            if depth >= max_depth:
                # Leaf
                split_indices.append(0)
                split_conditions.append(0.1 * (nid + 1) * ((-1) ** nid))
                left_children.append(-1)
                right_children.append(-1)
                default_left.append(1)
            else:
                feat = nid % n_features
                split_indices.append(feat)
                split_conditions.append(0.5 + nid * 0.1)
                left_children.append(2 * nid + 1)
                right_children.append(2 * nid + 2)
                default_left.append(1)

        trees.append({
            "tree_param": {"num_nodes": str(n_nodes)},
            "split_indices": split_indices,
            "split_conditions": split_conditions,
            "left_children": left_children,
            "right_children": right_children,
            "default_left": default_left,
        })

    return {
        "learner": {
            "learner_model_param": {
                "num_feature": str(n_features),
                "num_class": "0",
                "base_score": "0.5",
            },
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {"num_trees": str(n_trees)},
                    "trees": trees,
                    "tree_info": [0] * n_trees,
                },
                "gbtree_model_param": {"num_trees": str(n_trees)},
            },
            "objective": {"name": objective},
            "feature_names": [f"feat_{i}" for i in range(n_features)],
            "feature_types": ["float"] * n_features,
        },
        "version": [2, 0, 0],
    }


class TestXGBoostParser:
    def test_parse_regression(self):
        data = _make_xgboost_json(n_trees=3, max_depth=2, n_features=4)
        ir = _parse_xgboost_dict(data)

        assert ir.metadata.source_framework == "xgboost"
        ensemble = ir.get_tree_ensemble()
        assert ensemble is not None
        assert ensemble.n_trees == 3
        assert ensemble.n_features == 4
        assert ensemble.objective == Objective.REGRESSION

    def test_parse_binary_classification(self):
        data = _make_xgboost_json(objective="binary:logistic")
        ir = _parse_xgboost_dict(data)

        ensemble = ir.get_tree_ensemble()
        assert ensemble.objective == Objective.BINARY_CLASSIFICATION
        assert ensemble.n_classes == 2

    def test_tree_structure(self):
        data = _make_xgboost_json(n_trees=1, max_depth=2, n_features=3)
        ir = _parse_xgboost_dict(data)

        ensemble = ir.get_tree_ensemble()
        tree = ensemble.trees[0]
        assert tree.n_leaves > 0
        assert tree.n_internal > 0
        assert tree.max_depth == 2

        # Check that root is not a leaf
        root = tree.nodes[0]
        assert not root.is_leaf
        assert root.depth == 0

    def test_feature_names(self):
        data = _make_xgboost_json(n_features=5)
        ir = _parse_xgboost_dict(data)

        assert len(ir.metadata.feature_names) == 5
        assert ir.metadata.feature_names[0] == "feat_0"
        assert ir.schema.n_features == 5

    def test_file_parsing(self, tmp_path):
        data = _make_xgboost_json()
        model_file = tmp_path / "model.json"
        model_file.write_text(json.dumps(data))

        ir = parse_xgboost_json(model_file)
        assert ir.get_tree_ensemble() is not None

    def test_schema(self):
        data = _make_xgboost_json(n_features=3)
        ir = _parse_xgboost_dict(data)

        assert ir.schema.n_features == 3
        assert ir.schema.n_outputs >= 1
        assert ir.schema.input_fields[0].name == "feat_0"
        assert ir.schema.input_fields[0].dtype.value == "float32"

    def test_base_score(self):
        data = _make_xgboost_json()
        data["learner"]["learner_model_param"]["base_score"] = "0.25"
        ir = _parse_xgboost_dict(data)

        ensemble = ir.get_tree_ensemble()
        assert ensemble.base_score == pytest.approx(0.25)

    def test_parse_legacy_nodes_tree_format(self):
        # Older/alternate XGBoost JSON exports can represent trees via a "nodes" array.
        data = {
            "learner": {
                "learner_model_param": {
                    "num_feature": "2",
                    "num_class": "0",
                    "base_score": "0.5",
                },
                "gradient_booster": {
                    "model": {
                        "gbtree_model_param": {"num_trees": "1"},
                        "tree_info": [0],
                        "trees": [
                            {
                                "nodes": [
                                    {"split": 0, "split_condition": 0.25, "yes": 1, "no": 2, "missing": 1},
                                    {"leaf": -0.4},
                                    {"leaf": 0.7},
                                ]
                            }
                        ],
                    }
                },
                "objective": {"name": "binary:logistic"},
            },
            "version": [1, 7, 0],
        }

        ir = _parse_xgboost_dict(data)
        ensemble = ir.get_tree_ensemble()

        assert ensemble is not None
        assert ensemble.n_trees == 1
        assert ensemble.objective == Objective.BINARY_CLASSIFICATION
        assert ensemble.trees[0].n_internal == 1
        assert ensemble.trees[0].n_leaves == 2
