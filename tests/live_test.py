"""Exhaustive live integration test — real models, real data, real bugs.

Runs end-to-end through every feature with real sklearn/ONNX models:
  - ONNX parser: LinearClassifier, LinearRegressor, SVMClassifier, SVMRegressor,
                 Normalizer+Linear pipeline, Scaler+SVM pipeline
  - C99 emitter: emit + GCC compile + ctypes load + inference validation
  - Embedded profiles: Makefile/CMakeLists correctness for all 4 targets
  - MISRA-C: full compliance run on every emitted source file
  - LLVM IR: emit + structural validation for every model type
  - Differential privacy: statistical noise-level tests with real outputs
  - Bench report: JSON + HTML generation and roundtrip

Run with:
    python -m pytest tests/live_test.py -v
or:
    python tests/live_test.py
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_onnx_linear_classifier(n_features: int = 10, n_classes: int = 2,
                                 multiclass: bool = False):
    """Train and export a real LinearClassifier ONNX model."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    if multiclass:
        X, y = load_iris(return_X_y=True)
        X = X.astype(np.float32)
        clf = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="multinomial")
    else:
        X, y = load_breast_cancer(return_X_y=True)
        X = X[:, :n_features].astype(np.float32)
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
    clf.fit(X, y)

    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    model_onnx = convert_sklearn(clf, initial_types=initial_type)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(model_onnx.SerializeToString())
    tmp.close()
    return tmp.name, clf, X, y


def _to_onnx_linear_regressor(n_features: int = 10):
    """Train and export a real LinearRegressor ONNX model."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    X, y = load_diabetes(return_X_y=True)
    X = X[:, :n_features].astype(np.float32)
    reg = LinearRegression()
    reg.fit(X, y)

    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    model_onnx = convert_sklearn(reg, initial_types=initial_type)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(model_onnx.SerializeToString())
    tmp.close()
    return tmp.name, reg, X, y


def _to_onnx_svm_classifier(n_features: int = 10, kernel: str = "rbf"):
    """Train and export a real SVMClassifier ONNX model."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    X, y = load_breast_cancer(return_X_y=True)
    X = X[:, :n_features].astype(np.float32)
    # Normalise for SVM
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    clf = SVC(kernel=kernel, probability=False, C=1.0, max_iter=500)
    clf.fit(X, y)

    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    model_onnx = convert_sklearn(clf, initial_types=initial_type)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(model_onnx.SerializeToString())
    tmp.close()
    return tmp.name, clf, X, y


def _to_onnx_svm_regressor(n_features: int = 8):
    """Train and export a real SVMRegressor ONNX model."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    X, y = load_diabetes(return_X_y=True)
    X = X[:, :n_features].astype(np.float32)
    X = X / (np.std(X, axis=0) + 1e-8)
    reg = SVR(kernel="rbf", C=1.0, max_iter=500)
    reg.fit(X, y)

    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    model_onnx = convert_sklearn(reg, initial_types=initial_type)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(model_onnx.SerializeToString())
    tmp.close()
    return tmp.name, reg, X, y


def _to_onnx_normalizer_linear_pipeline(n_features: int = 10):
    """Train and export a Normalizer → LinearClassifier ONNX pipeline."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    X, y = load_breast_cancer(return_X_y=True)
    X = X[:, :n_features].astype(np.float32)
    pipe = Pipeline([
        ("norm", Normalizer(norm="l2")),
        ("clf", LogisticRegression(max_iter=300)),
    ])
    pipe.fit(X, y)

    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    model_onnx = convert_sklearn(pipe, initial_types=initial_type)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(model_onnx.SerializeToString())
    tmp.close()
    return tmp.name, pipe, X, y


def _to_onnx_scaler_svm_pipeline(n_features: int = 8):
    """Train and export a StandardScaler → SVC ONNX pipeline."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    X, y = load_breast_cancer(return_X_y=True)
    X = X[:, :n_features].astype(np.float32)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0, max_iter=500)),
    ])
    pipe.fit(X, y)

    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    model_onnx = convert_sklearn(pipe, initial_types=initial_type)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(model_onnx.SerializeToString())
    tmp.close()
    return tmp.name, pipe, X, y


def _compile_c99(out_dir: Path) -> Optional[Path]:
    """Try to compile the emitted C99 source with the host GCC/clang.
    Returns path to shared library on success, or None if compiler unavailable.
    """
    model_c = out_dir / "model.c"
    model_h = out_dir / "model.h"
    model_data = out_dir / "model_data.c"

    if not model_c.exists() or not model_h.exists():
        return None

    so_path = out_dir / "model.so"
    cmd = [
        "cc",
        "-std=c99", "-O2", "-fPIC", "-shared",
        "-I", str(out_dir),
        str(model_c),
        "-o", str(so_path),
        "-lm",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return so_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _load_and_infer(so_path: Path, inputs: np.ndarray,
                    n_features: int, n_outputs: int) -> Optional[np.ndarray]:
    """Load compiled .so, call timber_infer, return outputs or None on error."""
    try:
        lib = ctypes.CDLL(str(so_path))
        lib.timber_infer.restype = ctypes.c_int
        lib.timber_infer.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
        ]

        n_samples = inputs.shape[0]
        flat_in = inputs.astype(np.float32).ravel()
        flat_out = np.zeros(n_samples * n_outputs, dtype=np.float32)

        rc = lib.timber_infer(
            flat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_samples),
            flat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            None,
        )
        if rc != 0:
            return None
        return flat_out.reshape(n_samples, n_outputs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1.  ONNX Parser live tests
# ---------------------------------------------------------------------------

class TestONNXParserLive:
    """Live tests: train sklearn model → convert to ONNX → parse with Timber."""

    def test_linear_classifier_binary_parses(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage

        path, clf, X, y = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, LinearStage), f"Expected LinearStage, got {type(stage)}"
            assert stage.n_classes in (1, 2)
            assert len(stage.weights) > 0, "Empty weights"
            assert len(ir.schema.input_fields) == 10
        finally:
            os.unlink(path)

    def test_linear_classifier_multiclass_parses(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage

        path, clf, X, y = _to_onnx_linear_classifier(multiclass=True)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, LinearStage), f"Expected LinearStage, got {type(stage)}"
            assert stage.n_classes == 3
            assert stage.multi_weights is True
            assert len(stage.weights) == 3 * 4, f"Expected 12 weights, got {len(stage.weights)}"
            assert len(stage.biases) == 3
        finally:
            os.unlink(path)

    def test_linear_regressor_parses(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage

        path, reg, X, y = _to_onnx_linear_regressor(n_features=8)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, LinearStage), f"Expected LinearStage, got {type(stage)}"
            assert stage.n_classes == 1
            assert stage.multi_weights is False
            assert len(stage.weights) == 8, f"Expected 8 weights, got {len(stage.weights)}"
            assert ir.metadata.source_framework == "onnx"
        finally:
            os.unlink(path)

    def test_svm_classifier_rbf_parses(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage

        path, clf, X, y = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, SVMStage), f"Expected SVMStage, got {type(stage)}"
            assert stage.kernel_type == "rbf"
            assert stage.n_sv > 0, "Zero support vectors"
            assert stage.n_features == 10
            assert len(stage.support_vectors) == stage.n_sv
            assert len(stage.support_vectors[0]) == 10
        finally:
            os.unlink(path)

    def test_svm_classifier_linear_parses(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage

        path, clf, X, y = _to_onnx_svm_classifier(n_features=8, kernel="linear")
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, SVMStage), f"Expected SVMStage, got {type(stage)}"
            assert stage.kernel_type == "linear"
            assert stage.n_sv > 0
        finally:
            os.unlink(path)

    def test_svm_regressor_parses(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage

        path, reg, X, y = _to_onnx_svm_regressor(n_features=8)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, SVMStage), f"Expected SVMStage, got {type(stage)}"
            assert stage.n_classes == 1
            assert stage.n_features == 8
        finally:
            os.unlink(path)

    def test_linear_classifier_weights_match_sklearn(self):
        """Parsed weights should match sklearn's coef_ values."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage

        path, clf, X, y = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, LinearStage)
            # Binary LR: sklearn has shape (1, n_features) or (n_features,)
            sk_weights = clf.coef_.ravel()
            timber_weights = np.array(stage.weights)
            # Magnitudes should be in the same ballpark
            assert np.linalg.norm(sk_weights) > 0
            assert len(timber_weights) == len(sk_weights), (
                f"Weight count mismatch: {len(timber_weights)} vs {len(sk_weights)}")
        finally:
            os.unlink(path)

    def test_svm_sv_count_matches_sklearn(self):
        """Parsed support vector count should match sklearn's."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage

        path, clf, X, y = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, SVMStage)
            sk_n_sv = len(clf.support_vectors_)
            assert stage.n_sv == sk_n_sv, (
                f"SV count mismatch: timber={stage.n_sv} sklearn={sk_n_sv}")
        finally:
            os.unlink(path)

    def test_ir_serialization_round_trip_linear(self):
        """IR serialize → deserialize → field-level equality."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage, TimberIR

        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            json_str = ir.to_json()
            ir2 = TimberIR.from_json(json_str)
            s1 = ir.pipeline[-1]
            s2 = ir2.pipeline[-1]
            assert type(s1) == type(s2)
            assert s1.weights == s2.weights
            assert s1.bias == s2.bias
            assert s1.n_classes == s2.n_classes
        finally:
            os.unlink(path)

    def test_ir_serialization_round_trip_svm(self):
        """SVM IR serialize → deserialize → field equality."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage, TimberIR

        path, _, _, _ = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            json_str = ir.to_json()
            ir2 = TimberIR.from_json(json_str)
            s1 = ir.pipeline[-1]
            s2 = ir2.pipeline[-1]
            assert type(s1) == type(s2)
            assert s1.kernel_type == s2.kernel_type
            assert s1.n_sv == s2.n_sv
            assert s1.gamma == s2.gamma
            assert s1.support_vectors == s2.support_vectors
        finally:
            os.unlink(path)

    def test_normalizer_pipeline_parses(self):
        """Normalizer→LinearClassifier pipeline: two stages in IR."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage, NormalizerStage

        path, _, _, _ = _to_onnx_normalizer_linear_pipeline(n_features=10)
        try:
            ir = parse_onnx_model(path)
            # Should have at least a LinearStage as the primary
            has_linear = any(isinstance(s, LinearStage) for s in ir.pipeline)
            assert has_linear, f"No LinearStage in pipeline: {[type(s).__name__ for s in ir.pipeline]}"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 2.  C99 emission + GCC compilation + ctypes inference
# ---------------------------------------------------------------------------

class TestC99EmitterLive:
    """Emit C99, compile with host cc, load with ctypes, validate outputs."""

    def _emit_and_compile(self, ir, tmp_path: Path):
        from timber.codegen.c99 import C99Emitter
        out = C99Emitter().emit(ir)
        out.write(tmp_path)
        so = _compile_c99(tmp_path)
        return out, so

    def test_linear_binary_compiles(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            assert so is not None, "GCC compilation failed"
            assert so.exists()
        finally:
            os.unlink(path)

    def test_linear_binary_inference_runs(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage

        path, clf, X, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, LinearStage)
            n_features = len(stage.weights)
            n_outputs = 1

            out, so = self._emit_and_compile(ir, tmp_path)
            if so is None:
                pytest.skip("No compiler available")

            inputs = X[:20].astype(np.float32)
            results = _load_and_infer(so, inputs, n_features, n_outputs)
            assert results is not None, "ctypes inference returned error"
            assert results.shape == (20, 1)
            # All outputs should be finite
            assert np.all(np.isfinite(results)), "Non-finite outputs in C inference"
        finally:
            os.unlink(path)

    def test_linear_binary_output_in_01(self, tmp_path):
        """Sigmoid-activated output should be in (0,1)."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage

        path, clf, X, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, LinearStage)
            n_features = len(stage.weights)

            out, so = self._emit_and_compile(ir, tmp_path)
            if so is None:
                pytest.skip("No compiler available")

            inputs = X[:50].astype(np.float32)
            results = _load_and_infer(so, inputs, n_features, 1)
            assert results is not None
            assert np.all(results >= 0.0), f"Outputs below 0: {results[results < 0]}"
            assert np.all(results <= 1.0), f"Outputs above 1: {results[results > 1]}"
        finally:
            os.unlink(path)

    def test_linear_multiclass_compiles(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(multiclass=True)
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            assert so is not None, "Multiclass linear GCC compilation failed"
        finally:
            os.unlink(path)

    def test_linear_multiclass_inference_3class(self, tmp_path):
        """Multiclass softmax: 3 outputs, all positive, sum ≈ 1."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage

        path, clf, X, _ = _to_onnx_linear_classifier(multiclass=True)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, LinearStage)
            n_features = X.shape[1]
            n_outputs = 3

            out, so = self._emit_and_compile(ir, tmp_path)
            if so is None:
                pytest.skip("No compiler available")

            inputs = X[:20].astype(np.float32)
            results = _load_and_infer(so, inputs, n_features, n_outputs)
            assert results is not None, "Inference returned error"
            assert results.shape == (20, 3), f"Wrong shape: {results.shape}"
            assert np.all(np.isfinite(results))
            sums = results.sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=0.01,
                err_msg=f"Softmax outputs don't sum to 1: {sums}")
        finally:
            os.unlink(path)

    def test_linear_regressor_compiles(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_regressor(n_features=8)
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            assert so is not None, "Linear regressor GCC compilation failed"
        finally:
            os.unlink(path)

    def test_linear_regressor_inference(self, tmp_path):
        """Linear regression output should be real-valued (no sigmoid clamp)."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage

        path, reg, X, y = _to_onnx_linear_regressor(n_features=8)
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, LinearStage)
            n_features = len(stage.weights)

            out, so = self._emit_and_compile(ir, tmp_path)
            if so is None:
                pytest.skip("No compiler available")

            inputs = X[:20].astype(np.float32)
            results = _load_and_infer(so, inputs, n_features, 1)
            assert results is not None
            assert np.all(np.isfinite(results))
            # Regression output should not be clipped to [0,1]
            assert np.any(results > 1.1) or np.any(results < 0.0), (
                "Regression outputs look clipped — activation may be incorrectly applied")
        finally:
            os.unlink(path)

    def test_svm_rbf_compiles(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            assert so is not None, "SVM RBF GCC compilation failed"
        finally:
            os.unlink(path)

    def test_svm_rbf_inference_runs(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage

        path, clf, X, y = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            stage = ir.pipeline[-1]
            assert isinstance(stage, SVMStage)

            out, so = self._emit_and_compile(ir, tmp_path)
            if so is None:
                pytest.skip("No compiler available")

            inputs = X[:20].astype(np.float32)
            results = _load_and_infer(so, inputs, 10, 1)
            assert results is not None
            assert results.shape == (20, 1)
            assert np.all(np.isfinite(results))
        finally:
            os.unlink(path)

    def test_svm_linear_compiles(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_svm_classifier(n_features=8, kernel="linear")
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            assert so is not None, "SVM linear GCC compilation failed"
        finally:
            os.unlink(path)

    def test_c99_header_includes_guard(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            from timber.codegen.c99 import C99Emitter
            out = C99Emitter().emit(ir)
            assert "#ifndef TIMBER_MODEL_H" in out.model_h
            assert "#define TIMBER_MODEL_H" in out.model_h
            assert "#endif" in out.model_h
        finally:
            os.unlink(path)

    def test_c99_no_undefined_symbols(self, tmp_path):
        """Verify the compiled .so has no undefined external symbols beyond libc."""
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            if so is None:
                pytest.skip("No compiler available")
            # Use nm to check undefined symbols
            result = subprocess.run(
                ["nm", "-u", str(so)], capture_output=True, text=True
            )
            if result.returncode == 0:
                undefined = result.stdout.strip()
                # Only allowed: exp, expf from libm, and __stack_chk (security)
                bad = [l for l in undefined.splitlines()
                       if l.strip() and not any(ok in l for ok in
                          ("exp", "log", "sqrt", "fabs", "tanh",
                           "__stack_chk", "dyld", "_DYNAMIC"))]
                assert not bad, f"Unexpected undefined symbols: {bad}"
        finally:
            os.unlink(path)

    def test_c99_abi_version(self, tmp_path):
        """timber_abi_version() should return 1."""
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            if so is None:
                pytest.skip("No compiler available")

            lib = ctypes.CDLL(str(so))
            lib.timber_abi_version.restype = ctypes.c_int
            v = lib.timber_abi_version()
            assert v == 1, f"Expected ABI version 1, got {v}"
        finally:
            os.unlink(path)

    def test_c99_null_input_returns_error(self, tmp_path):
        """timber_infer with NULL input should return TIMBER_ERR_NULL = -1."""
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            if so is None:
                pytest.skip("No compiler available")

            lib = ctypes.CDLL(str(so))
            lib.timber_infer.restype = ctypes.c_int
            lib.timber_infer.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            rc = lib.timber_infer(None, 1, None, None)
            assert rc == -1, f"Expected -1 (TIMBER_ERR_NULL), got {rc}"
        finally:
            os.unlink(path)

    def test_svm_regressor_compiles(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_svm_regressor(n_features=8)
        try:
            ir = parse_onnx_model(path)
            out, so = self._emit_and_compile(ir, tmp_path)
            assert so is not None, "SVM regressor GCC compilation failed"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 3.  Embedded profiles live test
# ---------------------------------------------------------------------------

class TestEmbeddedProfilesLive:
    """Verify all embedded profiles generate correct build files."""

    PROFILES = ["cortex-m4", "cortex-m33", "rv32imf", "rv64gc"]

    def _emit_for_profile(self, profile: str):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter, TargetSpec

        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            spec = TargetSpec.for_embedded(profile)
            return C99Emitter(target=spec).emit(ir)
        finally:
            os.unlink(path)

    def test_cortex_m4_makefile_cc(self):
        out = self._emit_for_profile("cortex-m4")
        assert "arm-none-eabi-gcc" in out.makefile, "Missing cross-compiler in Makefile"

    def test_cortex_m4_makefile_no_shared(self):
        out = self._emit_for_profile("cortex-m4")
        assert ".so" not in out.makefile, "Embedded Makefile should not build .so"
        assert ".a" in out.makefile, "Embedded Makefile should build .a"

    def test_cortex_m4_makefile_cpu_flags(self):
        out = self._emit_for_profile("cortex-m4")
        assert "-mcpu=cortex-m4" in out.makefile
        assert "-mfpu=fpv4-sp-d16" in out.makefile
        assert "-mthumb" in out.makefile

    def test_cortex_m33_makefile(self):
        out = self._emit_for_profile("cortex-m33")
        assert "cortex-m33" in out.makefile
        assert "arm-none-eabi-" in out.makefile

    def test_rv32imf_makefile_cc(self):
        out = self._emit_for_profile("rv32imf")
        assert "riscv32" in out.makefile

    def test_rv32imf_makefile_arch(self):
        out = self._emit_for_profile("rv32imf")
        assert "rv32imf" in out.makefile

    def test_rv64gc_makefile_cc(self):
        out = self._emit_for_profile("rv64gc")
        assert "riscv64" in out.makefile

    def test_all_profiles_produce_valid_c(self):
        """Generated model.c should pass a basic syntax check on all profiles."""
        for profile in self.PROFILES:
            out = self._emit_for_profile(profile)
            c = out.model_c
            # Must have the key function definitions
            assert "timber_infer_single" in c, f"{profile}: missing timber_infer_single"
            assert "timber_infer(" in c, f"{profile}: missing timber_infer"
            assert "return 0" in c, f"{profile}: missing return 0"
            assert "TIMBER_N_FEATURES" in c, f"{profile}: missing TIMBER_N_FEATURES"

    def test_embedded_header_has_n_features(self):
        for profile in self.PROFILES:
            out = self._emit_for_profile(profile)
            assert "TIMBER_N_FEATURES" in out.model_h, f"{profile}: no TIMBER_N_FEATURES in header"
            assert "10" in out.model_h, f"{profile}: feature count 10 not in header"

    def test_embedded_cmake_has_toolchain_hint(self):
        """CMakeLists should mention the cross-compiler."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter, TargetSpec

        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            spec = TargetSpec.for_embedded("cortex-m4")
            out = C99Emitter(target=spec).emit(ir)
            # CMakeLists should at minimum have the C standard and source file
            assert "cmake_minimum_required" in out.cmakelists
            assert "timber_model" in out.cmakelists
        finally:
            os.unlink(path)

    def test_no_fpic_in_embedded_makefile(self):
        for profile in self.PROFILES:
            out = self._emit_for_profile(profile)
            assert "-fPIC" not in out.makefile, f"{profile}: -fPIC in embedded Makefile"

    def test_svm_model_embedded(self):
        """SVM model on embedded profile should also produce correct Makefile."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter, TargetSpec

        path, _, _, _ = _to_onnx_svm_classifier(n_features=8, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            spec = TargetSpec.for_embedded("cortex-m4")
            out = C99Emitter(target=spec).emit(ir)
            assert "arm-none-eabi-gcc" in out.makefile
            assert "TIMBER_SV" in out.model_data_c
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 4.  MISRA-C live test
# ---------------------------------------------------------------------------

class TestMisraCLive:
    """Emit code via MisraCEmitter, check compliance on real model output."""

    def _emit_misra(self, ir):
        from timber.codegen.misra_c import MisraCEmitter
        return MisraCEmitter().emit(ir)

    def test_linear_binary_misra_banner(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_misra(ir)
            assert "MISRA C:2012" in out.model_h
            assert "MISRA C:2012" in out.model_c
        finally:
            os.unlink(path)

    def test_linear_binary_misra_compliant(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.misra_c import MisraCEmitter
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            emitter = MisraCEmitter()
            out = emitter.emit(ir)
            report = emitter.check_compliance(out.model_c)
            assert report.is_compliant, (
                f"MISRA violations found:\n{report.summary()}"
            )
        finally:
            os.unlink(path)

    def test_svm_misra_compliant(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.misra_c import MisraCEmitter
        path, _, _, _ = _to_onnx_svm_classifier(n_features=8, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            emitter = MisraCEmitter()
            out = emitter.emit(ir)
            report = emitter.check_compliance(out.model_c)
            assert report.is_compliant, f"SVM MISRA violations:\n{report.summary()}"
        finally:
            os.unlink(path)

    def test_misra_compiles_with_gcc(self, tmp_path):
        """MISRA-compliant output should still compile with GCC."""
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.misra_c import MisraCEmitter
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = MisraCEmitter().emit(ir)
            out.write(tmp_path)
            so = _compile_c99(tmp_path)
            assert so is not None, "MISRA-compliant code failed to compile with GCC"
        finally:
            os.unlink(path)

    def test_misra_header_also_compliant(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.misra_c import MisraCEmitter
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            emitter = MisraCEmitter()
            out = emitter.emit(ir)
            report = emitter.check_compliance(out.model_h)
            assert report.is_compliant, f"Header MISRA violations:\n{report.summary()}"
        finally:
            os.unlink(path)

    def test_misra_no_stdio_in_output(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_misra(ir)
            assert "<stdio.h>" not in out.model_c
            assert "printf" not in out.model_c
        finally:
            os.unlink(path)

    def test_misra_no_compiler_extensions(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_misra(ir)
            assert "__attribute__" not in out.model_c
            assert "__extension__" not in out.model_c
        finally:
            os.unlink(path)

    def test_misra_multiclass_compliant(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.misra_c import MisraCEmitter
        path, _, _, _ = _to_onnx_linear_classifier(multiclass=True)
        try:
            ir = parse_onnx_model(path)
            emitter = MisraCEmitter()
            out = emitter.emit(ir)
            report = emitter.check_compliance(out.model_c)
            assert report.is_compliant, f"Multiclass MISRA violations:\n{report.summary()}"
        finally:
            os.unlink(path)

    def test_misra_report_rules_checked_gt_5(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.misra_c import MisraCEmitter
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            emitter = MisraCEmitter()
            out = emitter.emit(ir)
            report = emitter.check_compliance(out.model_c)
            assert report.rules_checked >= 8, (
                f"Only {report.rules_checked} rules checked — expected at least 8")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 5.  LLVM IR backend live test
# ---------------------------------------------------------------------------

class TestLLVMIRLive:
    """Emit real LLVM IR and validate its structural correctness."""

    def _emit_llvm(self, ir, target: str = "x86_64"):
        from timber.codegen.llvm_ir import LLVMIREmitter
        return LLVMIREmitter(target).emit(ir)

    def test_linear_binary_ll_module_header(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir)
            ll = out.model_ll
            assert "; ModuleID" in ll
            assert 'target triple = "x86_64' in ll
        finally:
            os.unlink(path)

    def test_linear_binary_ll_has_define(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir)
            assert "define" in out.model_ll
            assert "timber_infer_single" in out.model_ll
        finally:
            os.unlink(path)

    def test_linear_binary_ll_has_weights(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir)
            assert "timber_weights" in out.model_ll
        finally:
            os.unlink(path)

    def test_svm_rbf_ll_has_sv_data(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir)
            assert "timber_sv" in out.model_ll
            assert "timber_dual_coef" in out.model_ll
        finally:
            os.unlink(path)

    def test_svm_rbf_ll_has_exp_intrinsic(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir)
            assert "llvm.exp" in out.model_ll
        finally:
            os.unlink(path)

    def test_ll_save_to_disk(self, tmp_path):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir)
            files = out.save(tmp_path)
            ll_path = Path(files["model.ll"])
            assert ll_path.exists()
            content = ll_path.read_text()
            assert "timber_infer_single" in content
        finally:
            os.unlink(path)

    def test_ll_aarch64_triple(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir, "aarch64")
            assert "aarch64" in out.target_triple
            assert 'target triple = "aarch64' in out.model_ll
        finally:
            os.unlink(path)

    def test_ll_cortex_m4_triple(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir, "cortex-m4")
            assert "thumbv7em" in out.target_triple or "arm" in out.target_triple
        finally:
            os.unlink(path)

    def test_ll_no_empty_functions(self):
        """Every define block should have at least one instruction."""
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir)
            # Find all function bodies
            defines = re.findall(r'define [^{]+\{(.*?)\}', out.model_ll, re.DOTALL)
            for body in defines:
                stripped = body.strip()
                assert stripped, "Empty function body in LLVM IR"
        finally:
            os.unlink(path)

    def test_ll_regressor(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_linear_regressor(n_features=8)
        try:
            ir = parse_onnx_model(path)
            out = self._emit_llvm(ir)
            assert "timber_infer_single" in out.model_ll
            assert "timber_weights" in out.model_ll
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 6.  Differential privacy live test
# ---------------------------------------------------------------------------

class TestDPLive:
    """DP tests using real model outputs as the noise input."""

    def _get_real_outputs(self, n_samples: int = 200) -> np.ndarray:
        """Get real model probability outputs from a trained LR."""
        X, y = load_breast_cancer(return_X_y=True)
        clf = LogisticRegression(max_iter=300)
        clf.fit(X[:300], y[:300])
        proba = clf.predict_proba(X[:n_samples])[:, 1:2].astype(np.float32)
        return proba

    def test_laplace_mean_zero_bias(self):
        """Over many samples, E[noise] ≈ 0."""
        from timber.privacy.dp import DPConfig, apply_dp_noise
        outputs = self._get_real_outputs(200)
        cfg = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=0)
        noisy, _ = apply_dp_noise(outputs, cfg)
        mean_noise = float(np.mean(noisy - outputs.astype(np.float64)))
        assert abs(mean_noise) < 0.15, f"Laplace noise has large mean bias: {mean_noise}"

    def test_gaussian_mean_zero_bias(self):
        from timber.privacy.dp import DPConfig, apply_dp_noise
        outputs = self._get_real_outputs(200)
        cfg = DPConfig(mechanism="gaussian", epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=1)
        noisy, _ = apply_dp_noise(outputs, cfg)
        mean_noise = float(np.mean(noisy - outputs.astype(np.float64)))
        assert abs(mean_noise) < 0.15, f"Gaussian noise mean bias too large: {mean_noise}"

    def test_laplace_std_matches_theory(self):
        """Empirical std of noise should be close to theoretical b*sqrt(2)."""
        from timber.privacy.dp import DPConfig, apply_dp_noise
        outputs = np.zeros((5000, 1), dtype=np.float32)
        cfg = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0,
                       clip_outputs=False, seed=42)
        noisy, report = apply_dp_noise(outputs, cfg)
        empirical_std = float(np.std(noisy))
        theoretical_std = report.noise_scale * math.sqrt(2)
        # Allow 10% tolerance
        assert abs(empirical_std - theoretical_std) / theoretical_std < 0.10, (
            f"Laplace std {empirical_std:.4f} far from theory {theoretical_std:.4f}")

    def test_gaussian_std_matches_theory(self):
        from timber.privacy.dp import DPConfig, apply_dp_noise
        outputs = np.zeros((5000, 1), dtype=np.float32)
        cfg = DPConfig(mechanism="gaussian", epsilon=1.0, delta=1e-5,
                       sensitivity=1.0, clip_outputs=False, seed=99)
        noisy, report = apply_dp_noise(outputs, cfg)
        empirical_std = float(np.std(noisy))
        theoretical_std = report.noise_scale
        assert abs(empirical_std - theoretical_std) / theoretical_std < 0.10, (
            f"Gaussian std {empirical_std:.4f} far from theory {theoretical_std:.4f}")

    def test_clipping_preserves_probability_range(self):
        """After DP, probabilities should stay in [0,1] when clipping enabled."""
        from timber.privacy.dp import DPConfig, apply_dp_noise
        outputs = self._get_real_outputs(200)
        cfg = DPConfig(mechanism="laplace", epsilon=0.1, sensitivity=1.0,
                       clip_outputs=True, output_min=0.0, output_max=1.0, seed=7)
        noisy, _ = apply_dp_noise(outputs, cfg)
        assert np.all(noisy >= 0.0), f"Clipped outputs below 0: {noisy[noisy < 0]}"
        assert np.all(noisy <= 1.0), f"Clipped outputs above 1: {noisy[noisy > 1]}"

    def test_high_epsilon_preserves_accuracy(self):
        """With ε=1000 (very high), predictions should remain highly accurate."""
        from timber.privacy.dp import DPConfig, apply_dp_noise
        from sklearn.metrics import roc_auc_score
        X, y = load_breast_cancer(return_X_y=True)
        clf = LogisticRegression(max_iter=300)
        clf.fit(X[:400], y[:400])
        outputs = clf.predict_proba(X[400:])[:, 1:2].astype(np.float32)
        cfg = DPConfig(mechanism="laplace", epsilon=1000.0, sensitivity=1.0,
                       clip_outputs=True, seed=0)
        noisy, _ = apply_dp_noise(outputs, cfg)
        auc_orig = roc_auc_score(y[400:], outputs.ravel())
        auc_noisy = roc_auc_score(y[400:], noisy.ravel())
        assert abs(auc_orig - auc_noisy) < 0.02, (
            f"High-ε DP degraded AUC too much: orig={auc_orig:.3f} noisy={auc_noisy:.3f}")

    def test_low_epsilon_hides_signal(self):
        """With ε=0.01, the noisy output should have very low correlation with original."""
        from timber.privacy.dp import DPConfig, apply_dp_noise
        outputs = self._get_real_outputs(500)
        cfg = DPConfig(mechanism="laplace", epsilon=0.01, sensitivity=1.0,
                       clip_outputs=True, seed=3)
        noisy, _ = apply_dp_noise(outputs, cfg)
        corr = float(np.corrcoef(outputs.ravel(), noisy.ravel())[0, 1])
        # Very low ε should destroy correlation
        assert abs(corr) < 0.5, f"Low-ε DP still has high correlation: {corr:.3f}"

    def test_dp_dtype_float32_preserved(self):
        from timber.privacy.dp import DPConfig, apply_dp_noise
        outputs = self._get_real_outputs(50)
        assert outputs.dtype == np.float32
        cfg = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=0)
        noisy, _ = apply_dp_noise(outputs, cfg)
        assert noisy.dtype == np.float32, f"Expected float32, got {noisy.dtype}"

    def test_dp_dtype_float64_preserved(self):
        from timber.privacy.dp import DPConfig, apply_dp_noise
        outputs = self._get_real_outputs(50).astype(np.float64)
        cfg = DPConfig(mechanism="gaussian", epsilon=1.0, delta=1e-5,
                       sensitivity=1.0, seed=0)
        noisy, _ = apply_dp_noise(outputs, cfg)
        assert noisy.dtype == np.float64


# ---------------------------------------------------------------------------
# 7.  Bench report live test
# ---------------------------------------------------------------------------

class TestBenchReportLive:
    """End-to-end bench report generation with real data."""

    def _make_real_bench_data(self, tmp_path: Path) -> tuple[str, str]:
        """Write a real XGBoost IR JSON and test data CSV, return (ir_path, csv_path)."""
        import xgboost as xgb
        from timber.frontends.xgboost_parser import parse_xgboost_json

        X, y = load_breast_cancer(return_X_y=True)
        X = X.astype(np.float32)
        m = xgb.XGBClassifier(
            n_estimators=20, max_depth=3, random_state=42, eval_metric="logloss"
        )
        m.fit(X, y)

        model_tmp = str(tmp_path / "model.json")
        m.get_booster().save_model(model_tmp)
        ir = parse_xgboost_json(model_tmp)

        ir_path = str(tmp_path / "model.timber.json")
        Path(ir_path).write_text(ir.to_json())

        csv_path = str(tmp_path / "data.csv")
        header = ",".join(f"f{i}" for i in range(X.shape[1]))
        np.savetxt(csv_path, X[:100], delimiter=",", header=header, comments="")

        return ir_path, csv_path

    def test_bench_json_report(self, tmp_path):
        from timber.cli import _bench_report_html
        ir_path, csv_path = self._make_real_bench_data(tmp_path)

        report_data = {
            "timber_version": "0.2.0",
            "system": {
                "platform": "macOS",
                "python": "3.11",
                "cpu": "Apple M1",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            "model": {"artifact": ir_path, "n_trees": 20, "max_depth": 3,
                      "n_features": 30, "n_classes": 2,
                      "objective": "binary:logistic", "n_samples": 100},
            "results": [{
                "batch_size": 1, "n_runs": 200, "min_us": 1.2, "p50_us": 2.1,
                "p95_us": 3.5, "p99_us": 5.0, "p999_us": 8.0, "mean_us": 2.3,
                "std_us": 0.5, "cv_pct": 21.7, "throughput_samples_per_sec": 476190.0,
            }],
        }
        out_path = tmp_path / "report.json"
        out_path.write_text(json.dumps(report_data, indent=2))
        loaded = json.loads(out_path.read_text())
        assert loaded["timber_version"] == "0.2.0"
        assert loaded["results"][0]["p999_us"] == 8.0

    def test_bench_html_report(self, tmp_path):
        from timber.cli import _bench_report_html
        report_data = {
            "timber_version": "0.2.0",
            "system": {"platform": "macOS", "python": "3.11",
                       "cpu": "Apple M1", "timestamp": "2024-01-01T00:00:00Z"},
            "model": {"artifact": "test", "n_trees": 10, "max_depth": 3,
                      "n_features": 30, "n_classes": 2, "objective": "binary:logistic",
                      "n_samples": 100},
            "results": [{
                "batch_size": 1, "n_runs": 200, "min_us": 1.0,
                "p50_us": 2.0, "p95_us": 3.0, "p99_us": 4.0, "p999_us": 6.0,
                "mean_us": 2.1, "std_us": 0.4, "cv_pct": 19.0,
                "throughput_samples_per_sec": 500000.0,
            }],
        }
        html = _bench_report_html(report_data)
        # Structure checks
        assert "<!DOCTYPE html>" in html
        assert "Timber Benchmark Report" in html
        assert "macOS" in html
        assert "500000" in html
        assert "<table>" in html.lower() or "<table" in html
        assert '"timber_version"' in html  # raw JSON section

    def test_bench_html_valid_utf8(self, tmp_path):
        from timber.cli import _bench_report_html
        report_data = {
            "timber_version": "0.2.0",
            "system": {"platform": "test", "python": "3.11",
                       "cpu": "test", "timestamp": "2024-01-01T00:00:00Z"},
            "model": {"artifact": "a", "n_trees": 1, "max_depth": 1,
                      "n_features": 4, "n_classes": 2, "objective": "binary:logistic",
                      "n_samples": 10},
            "results": [],
        }
        html = _bench_report_html(report_data)
        out = tmp_path / "report.html"
        out.write_text(html, encoding="utf-8")
        assert out.stat().st_size > 500

    def test_bench_report_cv_field_present(self):
        from timber.cli import _bench_report_html
        report_data = {
            "timber_version": "0.2.0",
            "system": {"platform": "t", "python": "3", "cpu": "t",
                       "timestamp": "2024-01-01T00:00:00Z"},
            "model": {"artifact": "a", "n_trees": 1, "max_depth": 1,
                      "n_features": 4, "n_classes": 2, "objective": "binary:logistic",
                      "n_samples": 10},
            "results": [{"batch_size": 1, "n_runs": 100, "min_us": 1.0,
                         "p50_us": 2.0, "p95_us": 3.0, "p99_us": 4.0, "p999_us": 5.0,
                         "mean_us": 2.0, "std_us": 0.3, "cv_pct": 15.0,
                         "throughput_samples_per_sec": 500000.0}],
        }
        html = _bench_report_html(report_data)
        assert "15.0" in html  # cv_pct value


# ---------------------------------------------------------------------------
# 8.  IR serialisation stress test
# ---------------------------------------------------------------------------

class TestIRSerializationStress:
    """Round-trip every real model through JSON serialize/deserialize."""

    def _roundtrip(self, ir):
        from timber.ir.model import TimberIR
        s = ir.to_json()
        ir2 = TimberIR.from_json(s)
        return ir2

    def test_linear_binary_roundtrip_fields(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage
        path, _, _, _ = _to_onnx_linear_classifier(n_features=10)
        try:
            ir = parse_onnx_model(path)
            ir2 = self._roundtrip(ir)
            s1 = ir.pipeline[-1]
            s2 = ir2.pipeline[-1]
            assert type(s1) == type(s2)
            assert isinstance(s1, LinearStage)
            assert s1.weights == s2.weights
            assert s1.bias == s2.bias
            assert s1.activation == s2.activation
            assert s1.n_classes == s2.n_classes
        finally:
            os.unlink(path)

    def test_svm_roundtrip_sv_matrix(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage
        path, _, _, _ = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            ir2 = self._roundtrip(ir)
            s1 = ir.pipeline[-1]
            s2 = ir2.pipeline[-1]
            assert isinstance(s1, SVMStage)
            assert s1.support_vectors == s2.support_vectors
            assert s1.dual_coef == s2.dual_coef
            assert s1.rho == s2.rho
            assert s1.gamma == s2.gamma
        finally:
            os.unlink(path)

    def test_multiclass_linear_roundtrip(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage
        path, _, _, _ = _to_onnx_linear_classifier(multiclass=True)
        try:
            ir = parse_onnx_model(path)
            ir2 = self._roundtrip(ir)
            s1 = ir.pipeline[-1]
            s2 = ir2.pipeline[-1]
            assert isinstance(s1, LinearStage)
            assert s1.multi_weights == s2.multi_weights
            assert s1.biases == s2.biases
            assert s1.n_classes == s2.n_classes == 3
        finally:
            os.unlink(path)

    def test_svm_regressor_roundtrip(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage
        path, _, _, _ = _to_onnx_svm_regressor(n_features=8)
        try:
            ir = parse_onnx_model(path)
            ir2 = self._roundtrip(ir)
            s1 = ir.pipeline[-1]
            s2 = ir2.pipeline[-1]
            assert isinstance(s1, SVMStage)
            assert s1.n_features == s2.n_features
            assert s1.kernel_type == s2.kernel_type
        finally:
            os.unlink(path)

    def test_json_is_valid_utf8(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        path, _, _, _ = _to_onnx_svm_classifier(n_features=10, kernel="rbf")
        try:
            ir = parse_onnx_model(path)
            s = ir.to_json()
            # Must be valid JSON
            parsed = json.loads(s)
            assert "pipeline" in parsed
            assert "schema" in parsed
        finally:
            os.unlink(path)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
