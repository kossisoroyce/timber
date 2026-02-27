"""Rigorous tests: real framework models, numerical accuracy, large models, multi-class, edge cases."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.model_selection import train_test_split

from timber.frontends.xgboost_parser import parse_xgboost_json
from timber.frontends.lightgbm_parser import parse_lightgbm_model
from timber.optimizer.pipeline import OptimizerPipeline
from timber.codegen.c99 import C99Emitter, TargetSpec
from timber.runtime.predictor import TimberPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compile_and_predict(ir, X, tmp_path, optimize=False):
    """Compile IR to C, build shared lib, run predictions via ctypes."""
    if optimize:
        result = OptimizerPipeline().run(ir)
        ir = result.ir

    emitter = C99Emitter()
    output = emitter.emit(ir)
    output.write(tmp_path)

    predictor = TimberPredictor.from_artifact(tmp_path, build=True)
    preds = predictor.predict(X)
    predictor.close()
    return preds


# ---------------------------------------------------------------------------
# XGBoost — binary classification (Breast Cancer)
# ---------------------------------------------------------------------------

class TestXGBoostBinaryAccuracy:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        data = load_breast_cancer()
        self.X, self.y = data.data.astype(np.float32), data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.tmp_path = tmp_path

    def _train_and_save(self, n_estimators=50, max_depth=4):
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.1, objective="binary:logistic",
            random_state=42, eval_metric="logloss",
        )
        model.fit(self.X_train, self.y_train)
        path = self.tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        ref_probs = model.predict_proba(self.X_test)[:, 1]
        return path, ref_probs

    def test_exact_no_optimize(self):
        """Without optimization, predictions should match XGBoost within float32 tolerance."""
        path, ref_probs = self._train_and_save()
        ir = parse_xgboost_json(path)
        preds = _compile_and_predict(ir, self.X_test, self.tmp_path / "build", optimize=False)

        max_err = np.max(np.abs(ref_probs - preds))
        assert max_err < 1e-5, f"Max error {max_err:.2e} exceeds 1e-5"

    def test_with_optimization(self):
        """With optimization, predictions should still be very close."""
        path, ref_probs = self._train_and_save()
        ir = parse_xgboost_json(path)
        preds = _compile_and_predict(ir, self.X_test, self.tmp_path / "build", optimize=True)

        max_err = np.max(np.abs(ref_probs - preds))
        # Optimization may alter predictions slightly (dead leaf elimination)
        assert max_err < 0.05, f"Max error {max_err:.2e} exceeds 0.05"

    def test_large_model(self):
        """Large model: 200 trees, depth 6."""
        path, ref_probs = self._train_and_save(n_estimators=200, max_depth=6)
        ir = parse_xgboost_json(path)
        preds = _compile_and_predict(ir, self.X_test, self.tmp_path / "build", optimize=False)

        max_err = np.max(np.abs(ref_probs - preds))
        assert max_err < 1e-5, f"Max error {max_err:.2e} exceeds 1e-5"

    def test_shallow_model(self):
        """Shallow model: 10 trees, depth 2."""
        path, ref_probs = self._train_and_save(n_estimators=10, max_depth=2)
        ir = parse_xgboost_json(path)
        preds = _compile_and_predict(ir, self.X_test, self.tmp_path / "build", optimize=False)

        max_err = np.max(np.abs(ref_probs - preds))
        assert max_err < 1e-5, f"Max error {max_err:.2e} exceeds 1e-5"


# ---------------------------------------------------------------------------
# XGBoost — regression (Diabetes)
# ---------------------------------------------------------------------------

class TestXGBoostRegressionAccuracy:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        data = load_diabetes()
        self.X, self.y = data.data.astype(np.float32), data.target.astype(np.float32)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.tmp_path = tmp_path

    def test_regression_accuracy(self):
        model = xgb.XGBRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            objective="reg:squarederror", random_state=42,
        )
        model.fit(self.X_train, self.y_train)
        path = self.tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        ref_preds = model.predict(self.X_test)

        ir = parse_xgboost_json(path)
        preds = _compile_and_predict(ir, self.X_test, self.tmp_path / "build", optimize=False)

        max_err = np.max(np.abs(ref_preds - preds))
        mean_err = np.mean(np.abs(ref_preds - preds))
        assert max_err < 0.1, f"Max error {max_err:.4f} exceeds 0.1"
        assert mean_err < 0.01, f"Mean error {mean_err:.4f} exceeds 0.01"


# ---------------------------------------------------------------------------
# LightGBM — binary classification
# ---------------------------------------------------------------------------

class TestLightGBMBinaryAccuracy:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        data = load_breast_cancer()
        self.X, self.y = data.data.astype(np.float32), data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.tmp_path = tmp_path

    def test_lightgbm_binary(self):
        model = lgb.LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            objective="binary", random_state=42, verbose=-1,
        )
        model.fit(self.X_train, self.y_train)
        path = self.tmp_path / "model.txt"
        model.booster_.save_model(str(path))
        ref_probs = model.predict_proba(self.X_test)[:, 1]

        ir = parse_lightgbm_model(path)
        preds = _compile_and_predict(ir, self.X_test, self.tmp_path / "build", optimize=False)

        max_err = np.max(np.abs(ref_probs - preds))
        # LightGBM text parser may have slightly larger tolerance
        assert max_err < 0.01, f"Max error {max_err:.4f} exceeds 0.01"


# ---------------------------------------------------------------------------
# LightGBM — regression
# ---------------------------------------------------------------------------

class TestLightGBMRegressionAccuracy:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        data = load_diabetes()
        self.X, self.y = data.data.astype(np.float32), data.target.astype(np.float32)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.tmp_path = tmp_path

    def test_lightgbm_regression(self):
        model = lgb.LGBMRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            objective="regression", random_state=42, verbose=-1,
        )
        model.fit(self.X_train, self.y_train)
        path = self.tmp_path / "model.txt"
        model.booster_.save_model(str(path))
        ref_preds = model.predict(self.X_test)

        ir = parse_lightgbm_model(path)
        preds = _compile_and_predict(ir, self.X_test, self.tmp_path / "build", optimize=False)

        max_err = np.max(np.abs(ref_preds - preds))
        assert max_err < 1.0, f"Max error {max_err:.4f} exceeds 1.0"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_nan_inputs(self, tmp_path):
        """NaN in inputs should follow default_left path, not crash."""
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(
            n_estimators=20, max_depth=3, random_state=42, eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))

        ir = parse_xgboost_json(path)
        emitter = C99Emitter()
        output = emitter.emit(ir)
        output.write(tmp_path / "build")

        predictor = TimberPredictor.from_artifact(tmp_path / "build")

        # Inject NaN
        sample = X_test[0].copy()
        sample[0] = np.nan
        sample[5] = np.nan

        pred = predictor.predict_single(sample)
        assert np.isfinite(pred), "Prediction should be finite even with NaN inputs"
        assert 0.0 <= pred <= 1.0, "Binary classification prediction should be in [0, 1]"
        predictor.close()

    def test_single_tree(self, tmp_path):
        """Model with just 1 tree should work."""
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target

        model = xgb.XGBClassifier(
            n_estimators=1, max_depth=2, random_state=42, eval_metric="logloss",
        )
        model.fit(X, y)

        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))

        ir = parse_xgboost_json(path)
        ref_probs = model.predict_proba(X[:5])[:, 1]

        preds = _compile_and_predict(ir, X[:5].astype(np.float32), tmp_path / "build")
        max_err = np.max(np.abs(ref_probs - preds))
        assert max_err < 1e-4, f"Single tree max error {max_err:.2e}"

    def test_near_constant_predictions(self, tmp_path):
        """Near-constant target should produce near-constant predictions."""
        np.random.seed(42)
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.ones(100)
        y[:3] = 0  # need both classes for XGBoost

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=2, random_state=42, eval_metric="logloss",
        )
        model.fit(X, y)

        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))

        ir = parse_xgboost_json(path)
        preds = _compile_and_predict(ir, X[:10], tmp_path / "build")

        # Most predictions should be very close to 1.0
        assert np.mean(preds > 0.5) >= 0.8, f"Expected most preds > 0.5, got {np.mean(preds > 0.5):.2f}"

    def test_batch_consistency(self, tmp_path):
        """Batch predictions should equal single-sample predictions."""
        data = load_breast_cancer()
        X = data.data[:20].astype(np.float32)
        y = data.target

        model = xgb.XGBClassifier(
            n_estimators=30, max_depth=3, random_state=42, eval_metric="logloss",
        )
        model.fit(data.data.astype(np.float32), y)

        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))

        ir = parse_xgboost_json(path)
        emitter = C99Emitter()
        output = emitter.emit(ir)
        output.write(tmp_path / "build")

        predictor = TimberPredictor.from_artifact(tmp_path / "build")

        batch_preds = predictor.predict(X)
        single_preds = np.array([predictor.predict_single(X[i]) for i in range(len(X))])

        np.testing.assert_allclose(batch_preds, single_preds, rtol=1e-6)
        predictor.close()


# ---------------------------------------------------------------------------
# ABI and header validation
# ---------------------------------------------------------------------------

class TestGeneratedCodeQuality:
    def test_abi_version_present(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target

        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
        model.fit(X, y)

        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))

        ir = parse_xgboost_json(path)
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert "#define TIMBER_ABI_VERSION" in output.model_h
        assert '#define TIMBER_VERSION' in output.model_h
        assert "#define TIMBER_OK" in output.model_h
        assert "#define TIMBER_ERR_NULL" in output.model_h
        assert "#define TIMBER_ERR_BOUNDS" in output.model_h
        assert "timber_abi_version" in output.model_h
        assert "timber_abi_version" in output.model_c

    def test_double_accumulator(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target

        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
        model.fit(X, y)

        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))

        ir = parse_xgboost_json(path)
        emitter = C99Emitter()
        output = emitter.emit(ir)

        assert "double sum" in output.model_c

    def test_c99_compiles_clean(self, tmp_path):
        """Generated code should compile with -Wall -Werror."""
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="logloss")
        model.fit(X, y)

        path = tmp_path / "model.json"
        model.get_booster().save_model(str(path))

        ir = parse_xgboost_json(path)
        emitter = C99Emitter()
        output = emitter.emit(ir)
        output.write(tmp_path / "build")

        result = subprocess.run(
            ["gcc", "-std=c99", "-O2", "-Wall", "-Werror", "-fsyntax-only", "model.c"],
            capture_output=True, text=True, cwd=str(tmp_path / "build"),
        )
        assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"
