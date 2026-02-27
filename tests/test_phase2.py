"""Tests for Phase 2 components: WASM emitter, vectorization pass, serve endpoint, auto-detect."""

import json
import pickle
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

from timber.frontends.xgboost_parser import parse_xgboost_json
from timber.frontends.auto_detect import detect_format, parse_model
from timber.codegen.c99 import C99Emitter
from timber.codegen.wasm import WasmEmitter
from timber.optimizer.pipeline import OptimizerPipeline
from timber.optimizer.vectorize import vectorization_analysis, VectorizationHint


# ---------------------------------------------------------------------------
# WASM emitter
# ---------------------------------------------------------------------------

class TestWasmEmitter:
    def _make_ir(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
        model.fit(X, y)
        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        return parse_xgboost_json(path)

    def test_emit_wat(self, tmp_path):
        ir = self._make_ir(tmp_path)
        emitter = WasmEmitter()
        output = emitter.emit(ir)

        assert "(module" in output.wat
        assert "timber_infer_single" in output.wat
        assert "$traverse_tree" in output.wat
        assert "memory" in output.wat

    def test_emit_js_bindings(self, tmp_path):
        ir = self._make_ir(tmp_path)
        emitter = WasmEmitter()
        output = emitter.emit(ir)

        assert "loadTimberModel" in output.js_bindings
        assert "predict" in output.js_bindings
        assert "N_FEATURES" in output.js_bindings

    def test_write_files(self, tmp_path):
        ir = self._make_ir(tmp_path)
        emitter = WasmEmitter()
        output = emitter.emit(ir)
        files = output.write(tmp_path / "wasm_out")

        assert len(files) == 2
        assert any("model.wat" in f for f in files)
        assert any("timber_model.js" in f for f in files)

    def test_regression_model(self, tmp_path):
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X, y = data.data.astype(np.float32), data.target.astype(np.float32)
        model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
        model.fit(X, y)
        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        ir = parse_xgboost_json(path)

        emitter = WasmEmitter()
        output = emitter.emit(ir)
        # Regression should NOT have exp_neg / sigmoid
        assert "$exp_neg" not in output.wat

    def test_multiclass_model(self, tmp_path):
        data = load_iris()
        X, y = data.data.astype(np.float32), data.target
        model = xgb.XGBClassifier(
            n_estimators=5, max_depth=2, random_state=42,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        )
        model.fit(X, y)
        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        ir = parse_xgboost_json(path)

        emitter = WasmEmitter()
        output = emitter.emit(ir)
        assert "(module" in output.wat


# ---------------------------------------------------------------------------
# Vectorization analysis pass
# ---------------------------------------------------------------------------

class TestVectorizationPass:
    def _make_ir(self, tmp_path, n_estimators=20, max_depth=3):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, eval_metric="logloss",
        )
        model.fit(X, y)
        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        return parse_xgboost_json(path)

    def test_basic_analysis(self, tmp_path):
        ir = self._make_ir(tmp_path)
        result = vectorization_analysis(ir)

        assert result["trees_analyzed"] == 20
        assert "groups_found" in result
        assert "uniform_depth" in result
        assert "recommended_batch_tile" in result
        assert isinstance(result["hint"], VectorizationHint)

    def test_feature_frequencies(self, tmp_path):
        ir = self._make_ir(tmp_path)
        result = vectorization_analysis(ir)
        hint = result["hint"]

        assert len(hint.feature_frequencies) > 0
        assert hint.max_feature_accessed >= 0

    def test_feature_access_orders(self, tmp_path):
        ir = self._make_ir(tmp_path)
        result = vectorization_analysis(ir)
        hint = result["hint"]

        # Every tree should have an access order
        assert len(hint.feature_access_orders) == 20

    def test_pipeline_includes_vectorization(self, tmp_path):
        ir = self._make_ir(tmp_path)
        pipeline = OptimizerPipeline()
        opt_result = pipeline.run(ir)

        pass_names = [p.pass_name for p in opt_result.passes]
        assert "vectorization_analysis" in pass_names

    def test_annotations_stored(self, tmp_path):
        ir = self._make_ir(tmp_path)
        vectorization_analysis(ir)

        ensemble = ir.get_tree_ensemble()
        assert "vectorization_hint" in ensemble.annotations

    def test_shallow_trees_larger_tile(self, tmp_path):
        ir = self._make_ir(tmp_path, n_estimators=10, max_depth=2)
        result = vectorization_analysis(ir)
        assert result["recommended_batch_tile"] >= 4


# ---------------------------------------------------------------------------
# Auto-detect extended formats
# ---------------------------------------------------------------------------

class TestAutoDetectExtended:
    def test_detect_sklearn_pkl(self, tmp_path):
        p = tmp_path / "model.pkl"
        p.write_bytes(b"dummy")
        assert detect_format(str(p)) == "sklearn"

    def test_detect_sklearn_pickle(self, tmp_path):
        p = tmp_path / "model.pickle"
        p.write_bytes(b"dummy")
        assert detect_format(str(p)) == "sklearn"

    def test_detect_onnx(self, tmp_path):
        p = tmp_path / "model.onnx"
        p.write_bytes(b"dummy")
        assert detect_format(str(p)) == "onnx"

    def test_parse_model_sklearn(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        model = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42)
        model.fit(X, y)

        path = tmp_path / "model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)

        ir = parse_model(str(path))
        assert ir.metadata.source_framework == "sklearn"
        assert ir.get_tree_ensemble().n_trees == 5
