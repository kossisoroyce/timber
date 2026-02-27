"""End-to-end multi-class classification tests with real XGBoost and LightGBM models."""

import numpy as np
import pytest
import subprocess
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

from timber.frontends.xgboost_parser import parse_xgboost_json
from timber.frontends.lightgbm_parser import parse_lightgbm_model
from timber.optimizer.pipeline import OptimizerPipeline
from timber.codegen.c99 import C99Emitter
from timber.runtime.predictor import TimberPredictor


def _compile_and_predict_multi(ir, X, tmp_path, optimize=False):
    if optimize:
        ir = OptimizerPipeline().run(ir).ir
    emitter = C99Emitter()
    output = emitter.emit(ir)
    output.write(tmp_path)
    predictor = TimberPredictor.from_artifact(tmp_path, build=True)
    preds = predictor.predict(X)
    predictor.close()
    return preds


class TestXGBoostMulticlass:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path

    def test_iris_softprob(self):
        """3-class Iris with multi:softprob."""
        data = load_iris()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(
            n_estimators=30, max_depth=3, learning_rate=0.1,
            objective="multi:softprob", num_class=3,
            random_state=42, eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        path = self.tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        ref_probs = model.predict_proba(X_test)

        ir = parse_xgboost_json(path)
        ensemble = ir.get_tree_ensemble()
        assert ensemble.n_classes == 3
        assert ensemble.objective.value == "multi:softprob"

        preds = _compile_and_predict_multi(ir, X_test, self.tmp_path / "build")
        assert preds.shape == ref_probs.shape, f"Shape mismatch: {preds.shape} vs {ref_probs.shape}"

        max_err = np.max(np.abs(ref_probs - preds))
        mean_err = np.mean(np.abs(ref_probs - preds))
        assert max_err < 1e-4, f"Max error {max_err:.2e} exceeds 1e-4"
        assert mean_err < 1e-5, f"Mean error {mean_err:.2e} exceeds 1e-5"

        # Class predictions should match
        ref_classes = np.argmax(ref_probs, axis=1)
        pred_classes = np.argmax(preds, axis=1)
        agreement = np.mean(ref_classes == pred_classes)
        assert agreement == 1.0, f"Class agreement {agreement:.2%} < 100%"

    def test_wine_softprob(self):
        """3-class Wine with more features."""
        data = load_wine()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            objective="multi:softprob", num_class=3,
            random_state=42, eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        path = self.tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        ref_probs = model.predict_proba(X_test)

        ir = parse_xgboost_json(path)
        preds = _compile_and_predict_multi(ir, X_test, self.tmp_path / "build")

        max_err = np.max(np.abs(ref_probs - preds))
        assert max_err < 1e-4, f"Max error {max_err:.2e}"

    def test_iris_large_ensemble(self):
        """Large multi-class: 100 trees."""
        data = load_iris()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            objective="multi:softprob", num_class=3,
            random_state=42, eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        path = self.tmp_path / "model.json"
        model.get_booster().save_model(str(path))
        ref_probs = model.predict_proba(X_test)

        ir = parse_xgboost_json(path)
        preds = _compile_and_predict_multi(ir, X_test, self.tmp_path / "build")

        max_err = np.max(np.abs(ref_probs - preds))
        assert max_err < 1e-4, f"Max error {max_err:.2e}"

    def test_multiclass_compiles_clean(self):
        """Multi-class generated code should compile with -Wall -Werror."""
        data = load_iris()
        X, y = data.data.astype(np.float32), data.target

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=3,
            objective="multi:softprob", num_class=3,
            random_state=42, eval_metric="mlogloss",
        )
        model.fit(X, y)

        path = self.tmp_path / "model.json"
        model.get_booster().save_model(str(path))

        ir = parse_xgboost_json(path)
        emitter = C99Emitter()
        output = emitter.emit(ir)
        output.write(self.tmp_path / "build")

        result = subprocess.run(
            ["gcc", "-std=c99", "-O2", "-Wall", "-Werror", "-fsyntax-only", "model.c"],
            capture_output=True, text=True, cwd=str(self.tmp_path / "build"),
        )
        assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"


class TestLightGBMMulticlass:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path

    def test_iris_multiclass(self):
        """LightGBM 3-class Iris."""
        data = load_iris()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = lgb.LGBMClassifier(
            n_estimators=30, max_depth=3, learning_rate=0.1,
            objective="multiclass", num_class=3,
            random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)

        path = self.tmp_path / "model.txt"
        model.booster_.save_model(str(path))
        ref_probs = model.predict_proba(X_test)

        ir = parse_lightgbm_model(path)
        ensemble = ir.get_tree_ensemble()
        assert ensemble.n_classes == 3

        preds = _compile_and_predict_multi(ir, X_test, self.tmp_path / "build")
        assert preds.shape == ref_probs.shape

        max_err = np.max(np.abs(ref_probs - preds))
        assert max_err < 0.01, f"Max error {max_err:.4f}"
