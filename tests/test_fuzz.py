"""Fuzz tests for parsers — malformed, truncated, and adversarial inputs."""

import json
import pytest
from pathlib import Path

from timber.frontends.xgboost_parser import parse_xgboost_json, _parse_xgboost_dict
from timber.frontends.lightgbm_parser import parse_lightgbm_model
from timber.frontends.auto_detect import detect_format


class TestXGBoostFuzz:
    """XGBoost parser should raise clean errors, never crash."""

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("")
        with pytest.raises(Exception):
            parse_xgboost_json(p)

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json!!!")
        with pytest.raises(Exception):
            parse_xgboost_json(p)

    def test_empty_json_object(self, tmp_path):
        p = tmp_path / "empty_obj.json"
        p.write_text("{}")
        with pytest.raises(Exception):
            parse_xgboost_json(p)

    def test_missing_learner(self):
        with pytest.raises(Exception):
            _parse_xgboost_dict({"version": [2, 0, 0]})

    def test_missing_trees(self):
        data = {
            "learner": {
                "learner_model_param": {"num_feature": "5", "num_class": "0", "base_score": "0.5"},
                "objective": {"name": "reg:squarederror"},
                "gradient_booster": {"model": {"trees": [], "gbtree_model_param": {"num_trees": "0"}}},
            }
        }
        ir = _parse_xgboost_dict(data)
        ensemble = ir.get_tree_ensemble()
        assert ensemble.n_trees == 0

    def test_truncated_tree(self):
        """Tree with inconsistent node arrays."""
        data = {
            "learner": {
                "learner_model_param": {"num_feature": "2", "num_class": "0", "base_score": "0.5"},
                "objective": {"name": "reg:squarederror"},
                "gradient_booster": {
                    "model": {
                        "trees": [{
                            "split_indices": [0],
                            "split_conditions": [0.5],
                            "left_children": [-1],
                            "right_children": [-1],
                            "default_left": [1],
                            "tree_param": {"num_nodes": "1"},
                        }],
                        "gbtree_model_param": {"num_trees": "1"},
                        "tree_info": [0],
                    }
                },
            }
        }
        ir = _parse_xgboost_dict(data)
        assert ir.get_tree_ensemble().n_trees == 1

    def test_negative_feature_index(self):
        """Nodes referencing negative feature indices."""
        data = {
            "learner": {
                "learner_model_param": {"num_feature": "2", "num_class": "0", "base_score": "0.5"},
                "objective": {"name": "reg:squarederror"},
                "gradient_booster": {
                    "model": {
                        "trees": [{
                            "split_indices": [-1, 0, 0],
                            "split_conditions": [0.5, 1.0, 2.0],
                            "left_children": [1, -1, -1],
                            "right_children": [2, -1, -1],
                            "default_left": [1, 0, 0],
                            "tree_param": {"num_nodes": "3"},
                        }],
                        "gbtree_model_param": {"num_trees": "1"},
                        "tree_info": [0],
                    }
                },
            }
        }
        ir = _parse_xgboost_dict(data)
        assert ir.get_tree_ensemble().n_trees == 1

    def test_huge_threshold_values(self):
        """Extreme float values in thresholds."""
        data = {
            "learner": {
                "learner_model_param": {"num_feature": "2", "num_class": "0", "base_score": "0.5"},
                "objective": {"name": "reg:squarederror"},
                "gradient_booster": {
                    "model": {
                        "trees": [{
                            "split_indices": [0, 0, 0],
                            "split_conditions": [1e38, 1e-38, -1e38],
                            "left_children": [1, -1, -1],
                            "right_children": [2, -1, -1],
                            "default_left": [1, 0, 0],
                            "tree_param": {"num_nodes": "3"},
                        }],
                        "gbtree_model_param": {"num_trees": "1"},
                        "tree_info": [0],
                    }
                },
            }
        }
        ir = _parse_xgboost_dict(data)
        assert ir.get_tree_ensemble().n_trees == 1

    def test_zero_features(self):
        data = {
            "learner": {
                "learner_model_param": {"num_feature": "0", "num_class": "0", "base_score": "0.5"},
                "objective": {"name": "reg:squarederror"},
                "gradient_booster": {
                    "model": {
                        "trees": [],
                        "gbtree_model_param": {"num_trees": "0"},
                    }
                },
            }
        }
        ir = _parse_xgboost_dict(data)
        assert ir.get_tree_ensemble().n_features == 0

    def test_nan_base_score(self):
        data = {
            "learner": {
                "learner_model_param": {"num_feature": "2", "num_class": "0", "base_score": "NaN"},
                "objective": {"name": "reg:squarederror"},
                "gradient_booster": {
                    "model": {
                        "trees": [],
                        "gbtree_model_param": {"num_trees": "0"},
                    }
                },
            }
        }
        # Should not crash — NaN base_score is unusual but parsable
        ir = _parse_xgboost_dict(data)
        import math
        assert math.isnan(ir.get_tree_ensemble().base_score)


class TestLightGBMFuzz:
    """LightGBM parser should raise clean errors, never crash."""

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        with pytest.raises(Exception):
            parse_lightgbm_model(p)

    def test_garbage_content(self, tmp_path):
        p = tmp_path / "garbage.txt"
        p.write_text("this is not a lightgbm model at all\nrandom stuff\n")
        with pytest.raises(Exception):
            parse_lightgbm_model(p)

    def test_header_only(self, tmp_path):
        """A file with valid LightGBM header params but no trees produces an empty model."""
        p = tmp_path / "header_only.txt"
        p.write_text("tree\nversion=v3.3.5\nnum_class=1\nnum_tree_per_iteration=1\n")
        ir = parse_lightgbm_model(p)
        assert ir.get_tree_ensemble().n_trees == 0

    def test_binary_file(self, tmp_path):
        p = tmp_path / "binary.txt"
        p.write_bytes(b"\x00\x01\x02\xff\xfe\xfd" * 100)
        with pytest.raises(Exception):
            parse_lightgbm_model(p)


class TestAutoDetectFuzz:
    def test_nonexistent_file(self):
        result = detect_format("/nonexistent/path/model.xyz")
        assert result is None or isinstance(result, str)

    def test_unknown_extension(self, tmp_path):
        p = tmp_path / "model.xyz"
        p.write_text("hello")
        result = detect_format(str(p))
        # Should return None or best guess, not crash
        assert result is None or isinstance(result, str)

    def test_json_but_not_xgboost(self, tmp_path):
        p = tmp_path / "model.json"
        p.write_text('{"hello": "world"}')
        result = detect_format(str(p))
        # Should detect as json/xgboost based on extension, but parsing will fail gracefully
        assert isinstance(result, str) or result is None
