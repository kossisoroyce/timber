"""End-to-end integration tests for the full Timber compilation pipeline."""

import json
import numpy as np
import pytest
from pathlib import Path

from timber.frontends.xgboost_parser import _parse_xgboost_dict
from timber.frontends.lightgbm_parser import _parse_lightgbm_text
from timber.optimizer.pipeline import OptimizerPipeline
from timber.codegen.c99 import C99Emitter, TargetSpec
from timber.audit.report import AuditReport
from timber.ir.model import Objective


def _make_xgboost_model(n_trees=5, max_depth=3, n_features=4, objective="reg:squarederror"):
    """Create a synthetic XGBoost model dict."""
    trees = []
    for tid in range(n_trees):
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
                split_indices.append(0)
                split_conditions.append(0.05 * (nid + 1) * ((-1) ** (nid + tid)))
                left_children.append(-1)
                right_children.append(-1)
                default_left.append(1)
            else:
                feat = (nid + tid) % n_features
                split_indices.append(feat)
                split_conditions.append(0.5 + nid * 0.1 + tid * 0.01)
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
        },
        "version": [2, 0, 0],
    }


def _make_lightgbm_model_text(n_trees=3, n_features=4):
    """Create a synthetic LightGBM model text."""
    lines = [
        f"max_feature_idx={n_features - 1}",
        "num_class=1",
        "num_tree_per_iteration=1",
        "objective=regression",
        "learning_rate=0.1",
        "shrinkage_rate=0.1",
        f"feature_names={' '.join(f'feat_{i}' for i in range(n_features))}",
        "",
    ]

    for tid in range(n_trees):
        lines.append(f"Tree={tid}")
        lines.append("num_leaves=4")
        lines.append("num_cat=0")
        lines.append(f"split_feature={tid % n_features} {(tid + 1) % n_features} {(tid + 2) % n_features}")
        lines.append(f"threshold=0.5 0.3 0.7")
        lines.append("decision_type=2 2 2")
        lines.append("left_child=1 -1 -2")
        lines.append("right_child=2 -3 -4")
        lines.append(f"leaf_value={0.1 * (tid + 1)} {-0.05 * (tid + 1)} {0.2 * (tid + 1)} {-0.1 * (tid + 1)}")
        lines.append("")

    lines.append("end of trees")
    return "\n".join(lines)


class TestEndToEndXGBoost:
    def test_full_pipeline(self, tmp_path):
        """Parse -> Optimize -> Codegen -> Write for XGBoost."""
        # Parse
        data = _make_xgboost_model(n_trees=5, max_depth=3, n_features=4)
        ir = _parse_xgboost_dict(data)

        assert ir.get_tree_ensemble().n_trees == 5
        assert ir.get_tree_ensemble().n_features == 4

        # Optimize
        optimizer = OptimizerPipeline(dead_leaf_threshold=0.001)
        result = optimizer.run(ir)
        optimized_ir = result.ir

        assert optimized_ir.get_tree_ensemble() is not None
        assert len(result.passes) > 0

        # Codegen
        emitter = C99Emitter(target=TargetSpec(features=["avx2"]))
        output = emitter.emit(optimized_ir)

        # Write
        files = output.write(tmp_path / "dist")
        assert len(files) == 5
        for f in files:
            assert Path(f).exists()
            assert Path(f).stat().st_size > 0

        # Verify header content
        header = (tmp_path / "dist" / "model.h").read_text()
        assert "#define TIMBER_N_FEATURES 4" in header
        assert "#define TIMBER_N_TREES    5" in header

        # Verify model.c content
        model_c = (tmp_path / "dist" / "model.c").read_text()
        assert "timber_infer_single" in model_c
        assert "timber_infer" in model_c
        assert "traverse_tree" in model_c

        # Verify Makefile
        makefile = (tmp_path / "dist" / "Makefile").read_text()
        assert "-mavx2" in makefile

    def test_binary_classification_pipeline(self, tmp_path):
        """Full pipeline for binary classification with sigmoid."""
        data = _make_xgboost_model(n_trees=3, objective="binary:logistic")
        ir = _parse_xgboost_dict(data)

        optimizer = OptimizerPipeline()
        result = optimizer.run(ir)

        emitter = C99Emitter()
        output = emitter.emit(result.ir)
        files = output.write(tmp_path / "dist")

        model_c = (tmp_path / "dist" / "model.c").read_text()
        assert "exp(-sum)" in model_c

    def test_ir_serialization_roundtrip(self):
        """IR should survive JSON serialization."""
        data = _make_xgboost_model(n_trees=3)
        ir = _parse_xgboost_dict(data)

        json_str = ir.to_json()
        restored = ir.from_json(json_str)

        assert restored.get_tree_ensemble().n_trees == 3
        assert restored.schema.n_features == ir.schema.n_features

    def test_audit_report(self, tmp_path):
        """Audit report should be generated with all required fields."""
        data = _make_xgboost_model(n_trees=3)
        ir = _parse_xgboost_dict(data)

        optimizer = OptimizerPipeline()
        result = optimizer.run(ir)

        emitter = C99Emitter()
        output = emitter.emit(result.ir)
        files = output.write(tmp_path / "dist")

        report = AuditReport.generate(
            ir=result.ir,
            optimization_result=result,
            input_path="test_model.json",
            input_format="xgboost",
            target_spec={"arch": "x86_64"},
            output_files=files,
        )

        report_dict = report.to_dict()
        assert report_dict["timber_audit_report_version"] == "0.1"
        assert report_dict["compiler"]["timber_version"] == "0.1.0"
        assert "model_summary" in report_dict
        assert "optimization" in report_dict

        # Write and verify
        report_path = report.write(tmp_path / "dist" / "audit_report.json")
        assert Path(report_path).exists()
        loaded = json.loads(Path(report_path).read_text())
        assert loaded["timber_audit_report_version"] == "0.1"


class TestEndToEndLightGBM:
    def test_full_pipeline(self, tmp_path):
        """Parse -> Optimize -> Codegen for LightGBM."""
        text = _make_lightgbm_model_text(n_trees=3, n_features=4)
        ir = _parse_lightgbm_text(text)

        assert ir.metadata.source_framework == "lightgbm"
        assert ir.get_tree_ensemble().n_trees == 3

        optimizer = OptimizerPipeline()
        result = optimizer.run(ir)

        emitter = C99Emitter()
        output = emitter.emit(result.ir)
        files = output.write(tmp_path / "dist")

        assert len(files) == 5
        header = (tmp_path / "dist" / "model.h").read_text()
        assert "TIMBER_N_FEATURES" in header


class TestEndToEndWithCalibration:
    def test_calibration_branch_sort(self, tmp_path):
        """Branch sorting with calibration data."""
        data = _make_xgboost_model(n_trees=3, n_features=4)
        ir = _parse_xgboost_dict(data)

        calib = np.random.rand(100, 4).astype(np.float32)

        optimizer = OptimizerPipeline(calibration_data=calib)
        result = optimizer.run(ir)

        # Should have run the branch sort pass
        pass_names = [p.pass_name for p in result.passes]
        assert "frequency_branch_sort" in pass_names

        emitter = C99Emitter()
        output = emitter.emit(result.ir)
        output.write(tmp_path / "dist")
