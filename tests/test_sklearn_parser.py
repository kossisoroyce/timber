"""Tests for sklearn front-end parser — GradientBoosting, RandomForest, Pipeline."""

import pickle
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from timber.frontends.sklearn_parser import parse_sklearn_model, _convert_sklearn
from timber.codegen.c99 import C99Emitter
from timber.runtime.predictor import TimberPredictor


def _save_and_parse(model, tmp_path):
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return parse_sklearn_model(path)


def _compile_predict(ir, X, tmp_path):
    emitter = C99Emitter()
    output = emitter.emit(ir)
    output.write(tmp_path)
    predictor = TimberPredictor.from_artifact(tmp_path, build=True)
    preds = predictor.predict(X)
    predictor.close()
    return preds


class TestGradientBoostingClassifier:
    def test_binary_breast_cancer(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        ir = _save_and_parse(model, tmp_path)
        ensemble = ir.get_tree_ensemble()
        assert ensemble.n_trees == 20
        assert ensemble.is_boosted is True
        assert ir.metadata.source_framework == "sklearn"

    def test_binary_compiles_and_runs(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X_train, y_train)

        ir = _save_and_parse(model, tmp_path)
        preds = _compile_predict(ir, X_test[:5], tmp_path / "build")

        assert preds.shape == (5,)
        assert all(np.isfinite(preds))


class TestGradientBoostingRegressor:
    def test_diabetes_regression(self, tmp_path):
        data = load_diabetes()
        X, y = data.data.astype(np.float32), data.target.astype(np.float32)

        model = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X, y)

        ir = _save_and_parse(model, tmp_path)
        ensemble = ir.get_tree_ensemble()
        assert ensemble.n_trees == 20
        assert ensemble.objective.value == "reg:squarederror"

    def test_regression_compiles(self, tmp_path):
        data = load_diabetes()
        X, y = data.data.astype(np.float32), data.target.astype(np.float32)

        model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X, y)

        ir = _save_and_parse(model, tmp_path)
        preds = _compile_predict(ir, X[:5], tmp_path / "build")
        assert preds.shape == (5,)
        assert all(np.isfinite(preds))


class TestRandomForest:
    def test_rf_classifier(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target

        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        ir = _save_and_parse(model, tmp_path)
        ensemble = ir.get_tree_ensemble()
        assert ensemble.n_trees == 10
        assert ensemble.is_boosted is False

    def test_rf_regressor(self, tmp_path):
        data = load_diabetes()
        X, y = data.data.astype(np.float32), data.target.astype(np.float32)

        model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        ir = _save_and_parse(model, tmp_path)
        ensemble = ir.get_tree_ensemble()
        assert ensemble.n_trees == 10


class TestSklearnPipeline:
    def test_pipeline_with_scaler(self, tmp_path):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target

        pipe = SkPipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)),
        ])
        pipe.fit(X, y)

        ir = _save_and_parse(pipe, tmp_path)
        # Should have a scaler stage + tree ensemble stage
        assert len(ir.pipeline) == 2
        assert ir.pipeline[0].stage_type == "scaler"
        assert ir.pipeline[1].stage_type == "tree_ensemble"
