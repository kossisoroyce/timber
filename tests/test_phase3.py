"""Tests for Phase 3 components: CatBoost parser, stacking/voting, diff compilation, MISRA-C."""

import json
import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

from timber.frontends.xgboost_parser import parse_xgboost_json
from timber.frontends.catboost_parser import parse_catboost_json, _parse_catboost_dict
from timber.ir.ensemble_meta import (
    VotingEnsembleStage, StackingEnsembleStage,
    build_voting_ensemble, build_stacking_ensemble,
)
from timber.optimizer.diff_compile import diff_models, incremental_compile, DiffResult
from timber.codegen.misra_c import MisraCEmitter, MisraReport
from timber.codegen.c99 import C99Emitter


# ---------------------------------------------------------------------------
# CatBoost parser
# ---------------------------------------------------------------------------

class TestCatBoostParser:
    def _make_catboost_json(self, tmp_path, n_trees=3, depth=2, n_features=4):
        """Create a synthetic CatBoost JSON model for testing."""
        splits = []
        for d in range(depth):
            splits.append({
                "float_feature_index": d % n_features,
                "border": float(d + 1) * 0.5,
            })

        n_leaves = 1 << depth
        oblivious_trees = []
        for t in range(n_trees):
            leaf_values = [float(i + t) * 0.1 for i in range(n_leaves)]
            oblivious_trees.append({
                "splits": splits,
                "leaf_values": leaf_values,
            })

        data = {
            "model_info": {
                "params": {
                    "loss_function": {"type": "RMSE"},
                },
                "num_features": n_features,
            },
            "features_info": {
                "float_features": [
                    {"feature_index": i, "feature_name": f"feat_{i}"}
                    for i in range(n_features)
                ],
            },
            "oblivious_trees": oblivious_trees,
            "scale_and_bias": [[1.0], [0.0]],
        }

        path = tmp_path / "catboost_model.json"
        path.write_text(json.dumps(data))
        return path, data

    def test_parse_basic(self, tmp_path):
        path, _ = self._make_catboost_json(tmp_path)
        ir = parse_catboost_json(path)
        ensemble = ir.get_tree_ensemble()
        assert ensemble.n_trees == 3
        assert ir.metadata.source_framework == "catboost"

    def test_parse_tree_structure(self, tmp_path):
        path, _ = self._make_catboost_json(tmp_path, n_trees=1, depth=2)
        ir = parse_catboost_json(path)
        tree = ir.get_tree_ensemble().trees[0]
        # depth=2 -> 3 internal + 4 leaves = 7 nodes
        assert len(tree.nodes) == 7
        assert sum(1 for n in tree.nodes if n.is_leaf) == 4

    def test_parse_regression_objective(self, tmp_path):
        path, _ = self._make_catboost_json(tmp_path)
        ir = parse_catboost_json(path)
        assert ir.get_tree_ensemble().objective.value == "reg:squarederror"

    def test_parse_binary_classification(self, tmp_path):
        path, data = self._make_catboost_json(tmp_path)
        data["model_info"]["params"]["loss_function"] = {"type": "Logloss"}
        path.write_text(json.dumps(data))
        ir = parse_catboost_json(path)
        assert ir.get_tree_ensemble().objective.value == "binary:logistic"
        assert ir.get_tree_ensemble().n_classes == 2

    def test_compile_catboost_to_c99(self, tmp_path):
        path, _ = self._make_catboost_json(tmp_path, n_features=4)
        ir = parse_catboost_json(path)
        emitter = C99Emitter()
        output = emitter.emit(ir)
        assert "TIMBER_N_FEATURES" in output.model_h
        assert "traverse_tree" in output.model_c

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            _parse_catboost_dict({})

    def test_feature_names(self, tmp_path):
        path, _ = self._make_catboost_json(tmp_path, n_features=3)
        ir = parse_catboost_json(path)
        assert ir.metadata.feature_names == ["feat_0", "feat_1", "feat_2"]


# ---------------------------------------------------------------------------
# Stacking / Voting ensemble
# ---------------------------------------------------------------------------

class TestEnsembleMeta:
    def _make_two_irs(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        m1 = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
        m1.fit(X_train, y_train)
        p1 = tmp_path / "m1.json"
        m1.get_booster().save_model(str(p1))

        m2 = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=99, eval_metric="logloss")
        m2.fit(X_train, y_train)
        p2 = tmp_path / "m2.json"
        m2.get_booster().save_model(str(p2))

        ir1 = parse_xgboost_json(p1)
        ir2 = parse_xgboost_json(p2)
        return ir1, ir2

    def test_build_voting_ensemble(self, tmp_path):
        ir1, ir2 = self._make_two_irs(tmp_path)
        ve = build_voting_ensemble([ir1, ir2], voting="soft")
        assert len(ve.pipeline) == 1
        stage = ve.pipeline[0]
        assert isinstance(stage, VotingEnsembleStage)
        assert len(stage.sub_models) == 2
        assert stage.voting == "soft"

    def test_voting_weights(self, tmp_path):
        ir1, ir2 = self._make_two_irs(tmp_path)
        ve = build_voting_ensemble([ir1, ir2], weights=[0.7, 0.3])
        stage = ve.pipeline[0]
        assert stage.weights == [0.7, 0.3]

    def test_build_stacking_ensemble(self, tmp_path):
        ir1, ir2 = self._make_two_irs(tmp_path)
        se = build_stacking_ensemble([ir1, ir2], meta_model=ir1, passthrough=True)
        stage = se.pipeline[0]
        assert isinstance(stage, StackingEnsembleStage)
        assert stage.passthrough is True
        assert stage.meta_model is not None

    def test_empty_sub_models_raises(self):
        with pytest.raises(ValueError):
            build_voting_ensemble([])
        with pytest.raises(ValueError):
            build_stacking_ensemble([], meta_model=None)


# ---------------------------------------------------------------------------
# Differential compilation
# ---------------------------------------------------------------------------

class TestDiffCompile:
    def _make_ir(self, tmp_path, n_estimators=10, seed=42):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=2, random_state=seed, eval_metric="logloss")
        model.fit(X, y)
        path = tmp_path / f"model_{seed}.json"
        model.get_booster().save_model(str(path))
        return parse_xgboost_json(path)

    def test_identical_models(self, tmp_path):
        ir = self._make_ir(tmp_path)
        diff = diff_models(ir, ir)
        assert not diff.has_changes
        assert len(diff.unchanged_tree_ids) == 10

    def test_different_models(self, tmp_path):
        ir1 = self._make_ir(tmp_path, n_estimators=10, seed=42)
        ir2 = self._make_ir(tmp_path, n_estimators=10, seed=42)
        # Mutate a leaf value in ir2 to guarantee a difference
        ir2.get_tree_ensemble().trees[0].nodes[0].leaf_value += 999.0
        diff = diff_models(ir1, ir2)
        assert diff.has_changes
        assert len(diff.modified_tree_ids) > 0

    def test_added_trees(self, tmp_path):
        ir1 = self._make_ir(tmp_path, n_estimators=5, seed=42)
        ir2 = self._make_ir(tmp_path, n_estimators=10, seed=42)
        diff = diff_models(ir1, ir2)
        assert len(diff.added_tree_ids) == 5

    def test_removed_trees(self, tmp_path):
        ir1 = self._make_ir(tmp_path, n_estimators=10, seed=42)
        ir2 = self._make_ir(tmp_path, n_estimators=5, seed=42)
        diff = diff_models(ir1, ir2)
        assert len(diff.removed_tree_ids) == 5

    def test_incremental_compile(self, tmp_path):
        ir1 = self._make_ir(tmp_path, seed=42)
        ir2 = self._make_ir(tmp_path, seed=99)
        result = incremental_compile(ir1, ir2)
        ensemble = result.get_tree_ensemble()
        assert "diff" in ensemble.annotations
        assert ensemble.annotations["incremental"] is True

    def test_diff_summary(self, tmp_path):
        ir1 = self._make_ir(tmp_path, n_estimators=5, seed=42)
        ir2 = self._make_ir(tmp_path, n_estimators=10, seed=42)
        diff = diff_models(ir1, ir2)
        summary = diff.summary()
        assert "added" in summary
        assert "removed" in summary
        assert "modified" in summary
        assert "unchanged" in summary


# ---------------------------------------------------------------------------
# MISRA-C compliance
# ---------------------------------------------------------------------------

class TestMisraC:
    def _make_ir(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
        model.fit(X, y)
        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        return parse_xgboost_json(path)

    def test_misra_emitter_produces_output(self, tmp_path):
        ir = self._make_ir(tmp_path)
        emitter = MisraCEmitter()
        output = emitter.emit(ir)
        assert "MISRA" in output.model_h
        assert "MISRA" in output.model_c

    def test_compliance_check_clean(self, tmp_path):
        ir = self._make_ir(tmp_path)
        emitter = MisraCEmitter()
        output = emitter.emit(ir)
        report = emitter.check_compliance(output.model_c)
        assert isinstance(report, MisraReport)
        assert report.rules_checked > 0
        assert report.rules_passed > 0

    def test_compliance_check_detects_violations(self):
        emitter = MisraCEmitter()
        bad_code = '#define NULL ((void*)0)\n__attribute__((unused)) int foo;'
        report = emitter.check_compliance(bad_code)
        assert not report.is_compliant
        assert len(report.violations) > 0

    def test_misra_no_compiler_extensions(self, tmp_path):
        ir = self._make_ir(tmp_path)
        emitter = MisraCEmitter()
        output = emitter.emit(ir)
        assert "__attribute__" not in output.model_c
        assert "__extension__" not in output.model_c
