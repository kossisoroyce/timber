"""Nuclear-grade tests for the 5 new ML primitives:
  IsolationForest, OneClassSVM, Naive Bayes, GPR, k-NN.

Coverage per primitive:
  - IR dataclass instantiation and field correctness
  - JSON serialization round-trip (to_dict / from_dict)
  - sklearn → IR parsing (field values, shapes, types)
  - C99 header / data / inference structure checks
  - Compilation (if gcc available)
  - Numerical accuracy vs. sklearn reference (ctypes end-to-end)
  - End-to-end with an actual .pkl model file on disk
"""

from __future__ import annotations

import ctypes
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed=42):
    return np.random.default_rng(seed)


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


_skip_no_sklearn = pytest.mark.skipif(not _sklearn_available(), reason="scikit-learn not installed")


def _compile_so(build_dir: Path) -> Optional[Path]:
    """Compile model.c → libtimber_model.so; return path or None."""
    so = build_dir / "libtimber_model.so"
    ret = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-fPIC", "-shared",
         "-o", str(so), str(build_dir / "model.c"), "-lm"],
        capture_output=True,
    )
    if ret.returncode != 0:
        return None
    return so


def _ctypes_infer(so_path: Path, X: np.ndarray, n_outputs: int) -> Optional[np.ndarray]:
    """Load shared lib, run timber_infer on X, return [n_samples, n_outputs]."""
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError:
        return None
    lib.timber_infer.restype  = ctypes.c_int
    lib.timber_infer.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
    ]
    n = len(X)
    X32 = X.astype(np.float32, copy=False)
    out = np.zeros((n, n_outputs), dtype=np.float32)
    lib.timber_infer(
        X32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        None,
    )
    return out


# ============================================================
# _c_factor
# ============================================================

class TestCFactor:
    def test_c_factor_n1(self):
        from timber.ir.model import _c_factor
        assert _c_factor(1) == 1.0

    def test_c_factor_n2(self):
        from timber.ir.model import _c_factor
        assert _c_factor(2) == 2.0

    def test_c_factor_n10(self):
        from timber.ir.model import _c_factor
        import math
        expected = 2.0 * (math.log(9) + 0.5772156649) - 2.0 * 9 / 10
        assert abs(_c_factor(10) - expected) < 1e-9

    def test_c_factor_n256(self):
        from timber.ir.model import _c_factor
        v = _c_factor(256)
        assert 9.0 < v < 11.0  # sanity range

    def test_c_factor_increasing(self):
        from timber.ir.model import _c_factor
        vals = [_c_factor(n) for n in [2, 10, 50, 100, 256, 512]]
        assert all(vals[i] < vals[i+1] for i in range(len(vals)-1))


# ============================================================
# Isolation Forest
# ============================================================

class TestIsolationForestIR:
    def _make_stage(self):
        from timber.ir.model import IsolationForestStage, Tree, TreeNode
        tree = Tree(
            tree_id=0,
            nodes=[
                TreeNode(0, feature_index=1, threshold=0.5, left_child=1, right_child=2, is_leaf=False, leaf_value=0.0, depth=0),
                TreeNode(1, is_leaf=True, leaf_value=2.3, depth=1),
                TreeNode(2, is_leaf=True, leaf_value=1.1, depth=1),
            ],
            max_depth=1, n_leaves=2, n_internal=1,
        )
        return IsolationForestStage(
            stage_name="test_iforest", stage_type="isolation_forest",
            trees=[tree], n_features=4, max_samples=128, offset=-0.45,
        )

    def test_stage_type(self):
        s = self._make_stage()
        assert s.stage_type == "isolation_forest"

    def test_n_trees(self):
        s = self._make_stage()
        assert s.n_trees == 1

    def test_c_factor(self):
        from timber.ir.model import _c_factor
        s = self._make_stage()
        assert abs(s.c_factor - _c_factor(128)) < 1e-12

    def test_serialization_round_trip(self):
        import json
        from timber.ir.model import TimberIR, IsolationForestStage
        s = self._make_stage()
        ir = TimberIR(pipeline=[s])
        d = ir.to_dict()
        assert d["pipeline"][0]["stage_type"] == "isolation_forest"
        ir2 = TimberIR.from_dict(d)
        s2 = ir2.pipeline[0]
        assert isinstance(s2, IsolationForestStage)
        assert s2.max_samples == s.max_samples
        assert abs(s2.offset - s.offset) < 1e-9
        assert len(s2.trees) == 1
        assert len(s2.trees[0].nodes) == 3

    def test_leaf_values_preserved(self):
        from timber.ir.model import TimberIR
        s = self._make_stage()
        ir = TimberIR(pipeline=[s])
        ir2 = TimberIR.from_dict(ir.to_dict())
        leaves = [n for n in ir2.pipeline[0].trees[0].nodes if n.is_leaf]
        assert abs(leaves[0].leaf_value - 2.3) < 1e-5
        assert abs(leaves[1].leaf_value - 1.1) < 1e-5

    def test_offset_preserved_negative(self):
        from timber.ir.model import TimberIR
        s = self._make_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        assert abs(ir2.pipeline[0].offset - (-0.45)) < 1e-9


class TestIsolationForestParser:
    @pytest.fixture
    def fitted_model(self):
        from sklearn.ensemble import IsolationForest
        rng = _rng()
        X = rng.normal(size=(200, 5))
        clf = IsolationForest(n_estimators=20, max_samples=64, random_state=42)
        clf.fit(X)
        return clf, X

    @_skip_no_sklearn
    def test_parse_returns_correct_stage_type(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_isolation_forest
        from timber.ir.model import IsolationForestStage
        clf, _ = fitted_model
        stage = _parse_isolation_forest(clf)
        assert isinstance(stage, IsolationForestStage)

    @_skip_no_sklearn
    def test_parse_n_features(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_isolation_forest
        clf, _ = fitted_model
        stage = _parse_isolation_forest(clf)
        assert stage.n_features == 5

    @_skip_no_sklearn
    def test_parse_n_trees(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_isolation_forest
        clf, _ = fitted_model
        stage = _parse_isolation_forest(clf)
        assert stage.n_trees == 20

    @_skip_no_sklearn
    def test_parse_max_samples(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_isolation_forest
        clf, _ = fitted_model
        stage = _parse_isolation_forest(clf)
        assert stage.max_samples == 64

    @_skip_no_sklearn
    def test_parse_offset_matches_sklearn(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_isolation_forest
        clf, _ = fitted_model
        stage = _parse_isolation_forest(clf)
        assert abs(stage.offset - float(clf.offset_)) < 1e-6

    @_skip_no_sklearn
    def test_leaf_values_are_path_length_contributions(self, fitted_model):
        """All leaf values must be positive and bounded."""
        from timber.frontends.sklearn_parser import _parse_isolation_forest
        clf, _ = fitted_model
        stage = _parse_isolation_forest(clf)
        for tree in stage.trees:
            for node in tree.nodes:
                if node.is_leaf:
                    # min possible is depth=0 + c(1)=1 = 1.0
                    assert node.leaf_value >= 1.0, f"Negative/zero path length: {node.leaf_value}"
                    # max reasonable: depth ~12 + c(64) ~8 ≈ 20
                    assert node.leaf_value < 30.0, f"Unreasonably large: {node.leaf_value}"

    @_skip_no_sklearn
    def test_feature_indices_in_range(self, fitted_model):
        """Feature indices must be valid original feature indices."""
        from timber.frontends.sklearn_parser import _parse_isolation_forest
        clf, _ = fitted_model
        stage = _parse_isolation_forest(clf)
        for tree in stage.trees:
            for node in tree.nodes:
                if not node.is_leaf:
                    assert 0 <= node.feature_index < 5

    @_skip_no_sklearn
    def test_full_parse_via_convert_sklearn(self, fitted_model):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import IsolationForestStage
        clf, _ = fitted_model
        ir = _convert_sklearn(clf)
        assert isinstance(ir.pipeline[0], IsolationForestStage)
        assert ir.schema.n_features == 5
        assert ir.schema.n_outputs == 1


class TestIsolationForestC99:
    @pytest.fixture
    def ir(self):
        from sklearn.ensemble import IsolationForest
        from timber.frontends.sklearn_parser import _parse_isolation_forest
        from timber.ir.model import TimberIR, Schema, Field, FieldType, Metadata
        rng = _rng()
        X = rng.normal(size=(100, 4))
        clf = IsolationForest(n_estimators=10, max_samples=50, random_state=0)
        clf.fit(X)
        stage = _parse_isolation_forest(clf)
        ir = TimberIR(
            pipeline=[stage],
            schema=Schema(
                input_fields=[Field(f"f{i}", FieldType.FLOAT32, i) for i in range(4)],
                output_fields=[Field("score", FieldType.FLOAT32, 0)],
            ),
            metadata=Metadata(source_framework="sklearn"),
        )
        return ir, clf, X

    @_skip_no_sklearn
    def test_emit_produces_c99output(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        out = C99Emitter().emit(ir_obj)
        assert out.model_c
        assert out.model_h
        assert out.model_data_c

    @_skip_no_sklearn
    def test_header_contains_n_features(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        h = C99Emitter().emit(ir_obj).model_h
        assert "TIMBER_N_FEATURES 4" in h

    @_skip_no_sklearn
    def test_header_contains_n_outputs_1(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        h = C99Emitter().emit(ir_obj).model_h
        assert "TIMBER_N_OUTPUTS  1" in h

    @_skip_no_sklearn
    def test_model_c_contains_exp2(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        c = C99Emitter().emit(ir_obj).model_c
        assert "exp2(" in c

    @_skip_no_sklearn
    def test_model_data_contains_offset(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        dc = C99Emitter().emit(ir_obj).model_data_c
        assert "TIMBER_IF_OFFSET" in dc
        assert "TIMBER_C_MAX" in dc

    @_skip_no_sklearn
    def test_compiles(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        assert so.exists()

    @_skip_no_sklearn
    def test_numerical_accuracy_vs_sklearn(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, clf, X = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        result = _ctypes_infer(so, X[:50].astype(np.float32), 1)
        assert result is not None
        ref = clf.decision_function(X[:50].astype(np.float32))
        timber_vals = result.flatten().astype(np.float64)
        # Tolerance: float32 path lengths → ~0.01 absolute error on anomaly score
        np.testing.assert_allclose(timber_vals, ref, atol=0.05, rtol=0.1,
                                   err_msg="IsolationForest decision_function mismatch vs sklearn")

    @_skip_no_sklearn
    def test_sign_agreement_vs_sklearn(self, ir, tmp_path):
        """Sign of decision_function must agree on majority of samples."""
        from timber.codegen.c99 import C99Emitter
        ir_obj, clf, X = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        result = _ctypes_infer(so, X[:100].astype(np.float32), 1)
        assert result is not None
        ref  = clf.decision_function(X[:100])
        c99  = result.flatten()
        sign_match = np.mean(np.sign(ref) == np.sign(c99))
        assert sign_match >= 0.88, f"Sign agreement too low: {sign_match:.2%}"


class TestIsolationForestEndToEnd:
    @_skip_no_sklearn
    def test_pkl_round_trip(self, tmp_path):
        from sklearn.ensemble import IsolationForest
        from timber.frontends.sklearn_parser import parse_sklearn_model
        from timber.ir.model import IsolationForestStage
        rng = _rng()
        X = rng.normal(size=(150, 6))
        clf = IsolationForest(n_estimators=15, max_samples=80, random_state=7)
        clf.fit(X)
        pkl = tmp_path / "iforest.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(clf, f)
        ir = parse_sklearn_model(str(pkl))
        assert isinstance(ir.pipeline[0], IsolationForestStage)
        assert ir.pipeline[0].n_features == 6
        assert ir.pipeline[0].n_trees == 15

    @_skip_no_sklearn
    def test_pkl_compile_and_infer(self, tmp_path):
        from sklearn.ensemble import IsolationForest
        from timber.frontends.sklearn_parser import parse_sklearn_model
        from timber.codegen.c99 import C99Emitter
        rng = _rng()
        X = rng.normal(size=(200, 4))
        clf = IsolationForest(n_estimators=10, max_samples=64, random_state=1)
        clf.fit(X)
        pkl = tmp_path / "iforest.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(clf, f)
        ir = parse_sklearn_model(str(pkl))
        build = tmp_path / "build"
        C99Emitter().emit(ir).write(build)
        so = _compile_so(build)
        if so is None:
            pytest.skip("no gcc")
        X_test = rng.normal(size=(20, 4)).astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        assert result is not None
        ref = clf.decision_function(X_test)
        np.testing.assert_allclose(result.flatten(), ref, atol=0.05, rtol=0.1)


# ============================================================
# One-Class SVM
# ============================================================

class TestOneClassSVMIR:
    def _make_stage(self):
        from timber.ir.model import SVMStage, Objective
        return SVMStage(
            stage_name="test_ocsvm", stage_type="svm",
            kernel_type="rbf",
            support_vectors=[[1.0, 2.0], [3.0, 4.0]],
            dual_coef=[0.5, 0.5],
            rho=[-0.1],
            n_support=[2],
            gamma=0.5,
            coef0=0.0,
            degree=3,
            n_features=2,
            n_classes=1,
            objective=Objective.CUSTOM,
            is_one_class=True,
        )

    def test_is_one_class_flag(self):
        s = self._make_stage()
        assert s.is_one_class is True

    def test_serialization_preserves_is_one_class(self):
        from timber.ir.model import TimberIR, SVMStage
        s = self._make_stage()
        ir = TimberIR(pipeline=[s])
        ir2 = TimberIR.from_dict(ir.to_dict())
        s2 = ir2.pipeline[0]
        assert isinstance(s2, SVMStage)
        assert s2.is_one_class is True

    def test_regular_svm_is_one_class_false_by_default(self):
        from timber.ir.model import SVMStage, Objective
        s = SVMStage(stage_name="x", stage_type="svm",
                     kernel_type="rbf", support_vectors=[], dual_coef=[],
                     rho=[0.0], n_support=[], gamma=1.0, coef0=0.0, degree=3,
                     n_features=4, n_classes=2, objective=Objective.BINARY_CLASSIFICATION)
        assert s.is_one_class is False

    def test_round_trip_all_fields(self):
        from timber.ir.model import TimberIR, SVMStage
        s = self._make_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        assert s2.n_classes == 1
        assert s2.kernel_type == "rbf"
        assert abs(s2.gamma - 0.5) < 1e-9
        assert abs(s2.rho[0] - (-0.1)) < 1e-9


class TestOneClassSVMParser:
    @pytest.fixture
    def fitted_model(self):
        from sklearn.svm import OneClassSVM
        rng = _rng()
        X = rng.normal(size=(100, 4))
        clf = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05)
        clf.fit(X)
        return clf, X

    @_skip_no_sklearn
    def test_parse_is_one_class(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_one_class_svm
        clf, _ = fitted_model
        stage = _parse_one_class_svm(clf)
        assert stage.is_one_class is True

    @_skip_no_sklearn
    def test_parse_n_classes_is_1(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_one_class_svm
        clf, _ = fitted_model
        stage = _parse_one_class_svm(clf)
        assert stage.n_classes == 1

    @_skip_no_sklearn
    def test_parse_n_features(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_one_class_svm
        clf, _ = fitted_model
        stage = _parse_one_class_svm(clf)
        assert stage.n_features == 4

    @_skip_no_sklearn
    def test_parse_n_sv_matches_sklearn(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_one_class_svm
        clf, _ = fitted_model
        stage = _parse_one_class_svm(clf)
        assert stage.n_sv == clf.support_vectors_.shape[0]

    @_skip_no_sklearn
    def test_parse_gamma_matches(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_one_class_svm
        clf, _ = fitted_model
        stage = _parse_one_class_svm(clf)
        assert abs(stage.gamma - float(clf._gamma)) < 1e-7

    @_skip_no_sklearn
    def test_rho_sign_convention(self, fitted_model):
        """rho stores intercept_[0] directly so C99 decision = rho + sum(kernel) matches sklearn."""
        from timber.frontends.sklearn_parser import _parse_one_class_svm
        clf, _ = fitted_model
        stage = _parse_one_class_svm(clf)
        assert abs(stage.rho[0] - float(clf.intercept_[0])) < 1e-7

    @_skip_no_sklearn
    def test_convert_sklearn_dispatch(self, fitted_model):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import SVMStage
        clf, _ = fitted_model
        ir = _convert_sklearn(clf)
        s = ir.pipeline[0]
        assert isinstance(s, SVMStage)
        assert s.is_one_class is True


class TestOneClassSVMC99:
    @pytest.fixture
    def ir(self):
        from sklearn.svm import OneClassSVM
        from timber.frontends.sklearn_parser import _parse_one_class_svm
        from timber.ir.model import TimberIR, Schema, Field, FieldType, Metadata
        rng = _rng()
        X = rng.normal(size=(80, 3))
        clf = OneClassSVM(kernel="rbf", gamma=0.5, nu=0.1)
        clf.fit(X)
        stage = _parse_one_class_svm(clf)
        ir = TimberIR(
            pipeline=[stage],
            schema=Schema(
                input_fields=[Field(f"f{i}", FieldType.FLOAT32, i) for i in range(3)],
                output_fields=[Field("score", FieldType.FLOAT32, 0)],
            ),
            metadata=Metadata(source_framework="sklearn"),
        )
        return ir, clf, X

    @_skip_no_sklearn
    def test_emit_produces_output(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        out = C99Emitter().emit(ir_obj)
        assert "timber_infer_single" in out.model_c

    @_skip_no_sklearn
    def test_header_n_outputs_1(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        h = C99Emitter().emit(ir_obj).model_h
        assert "TIMBER_N_OUTPUTS  1" in h

    @_skip_no_sklearn
    def test_compiles(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        assert so.exists()

    @_skip_no_sklearn
    def test_numerical_accuracy_vs_sklearn(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, clf, X = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        X_test = X[:30].astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        assert result is not None
        ref = clf.decision_function(X_test).flatten()
        np.testing.assert_allclose(result.flatten(), ref, atol=1e-3, rtol=1e-3)

    @_skip_no_sklearn
    def test_sign_agreement_with_sklearn(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, clf, X = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        result = _ctypes_infer(so, X.astype(np.float32), 1)
        ref = clf.decision_function(X)
        assert np.mean(np.sign(result.flatten()) == np.sign(ref)) >= 0.98


class TestOneClassSVMEndToEnd:
    @_skip_no_sklearn
    def test_pkl_compile_and_match(self, tmp_path):
        from sklearn.svm import OneClassSVM
        from timber.frontends.sklearn_parser import parse_sklearn_model
        from timber.codegen.c99 import C99Emitter
        rng = _rng()
        X = rng.normal(size=(120, 5))
        clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        clf.fit(X)
        pkl = tmp_path / "ocsvm.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(clf, f)
        ir = parse_sklearn_model(str(pkl))
        build = tmp_path / "build"
        C99Emitter().emit(ir).write(build)
        so = _compile_so(build)
        if so is None:
            pytest.skip("no gcc")
        X_test = rng.normal(size=(25, 5)).astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        ref = clf.decision_function(X_test).flatten()
        np.testing.assert_allclose(result.flatten(), ref, atol=1e-3, rtol=1e-3)


# ============================================================
# Naive Bayes
# ============================================================

class TestNaiveBayesIR:
    def _make_stage(self):
        import math
        from timber.ir.model import NaiveBayesStage
        TWO_PI = 2.0 * math.pi
        theta    = [[0.0, 1.0], [1.0, 0.0]]
        var      = [[1.0, 1.0], [1.0, 1.0]]
        log_vc   = [[-0.5 * math.log(TWO_PI * v) for v in row] for row in var]
        inv_2v   = [[1.0 / (2.0 * v) for v in row] for row in var]
        return NaiveBayesStage(
            stage_name="test_nb", stage_type="naive_bayes",
            log_prior=[math.log(0.5), math.log(0.5)],
            theta=theta,
            log_var_const=log_vc,
            inv_2var=inv_2v,
            n_classes=2,
            n_features=2,
        )

    def test_stage_type(self):
        assert self._make_stage().stage_type == "naive_bayes"

    def test_n_classes_n_features(self):
        s = self._make_stage()
        assert s.n_classes == 2
        assert s.n_features == 2

    def test_serialization_round_trip(self):
        from timber.ir.model import TimberIR, NaiveBayesStage
        s = self._make_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        assert isinstance(s2, NaiveBayesStage)
        assert s2.n_classes == 2
        assert len(s2.theta) == 2
        assert len(s2.theta[0]) == 2

    def test_theta_preserved(self):
        from timber.ir.model import TimberIR
        s = self._make_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        assert abs(s2.theta[0][1] - 1.0) < 1e-9
        assert abs(s2.theta[1][0] - 1.0) < 1e-9

    def test_log_prior_sum_to_log1(self):
        import math
        s = self._make_stage()
        total = sum(math.exp(lp) for lp in s.log_prior)
        assert abs(total - 1.0) < 1e-9


class TestNaiveBayesParser:
    @pytest.fixture
    def fitted_model(self):
        from sklearn.naive_bayes import GaussianNB
        rng = _rng()
        X = rng.normal(size=(200, 6))
        y = (X[:, 0] > 0).astype(int)
        clf = GaussianNB()
        clf.fit(X, y)
        return clf, X, y

    @_skip_no_sklearn
    def test_parse_returns_nb_stage(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_naive_bayes
        from timber.ir.model import NaiveBayesStage
        clf, _, _ = fitted_model
        stage = _parse_naive_bayes(clf)
        assert isinstance(stage, NaiveBayesStage)

    @_skip_no_sklearn
    def test_parse_n_features(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_naive_bayes
        clf, _, _ = fitted_model
        assert _parse_naive_bayes(clf).n_features == 6

    @_skip_no_sklearn
    def test_parse_n_classes(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_naive_bayes
        clf, _, _ = fitted_model
        assert _parse_naive_bayes(clf).n_classes == 2

    @_skip_no_sklearn
    def test_theta_matches_sklearn(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_naive_bayes
        clf, _, _ = fitted_model
        stage = _parse_naive_bayes(clf)
        np.testing.assert_allclose(
            np.array(stage.theta), clf.theta_, rtol=1e-5,
            err_msg="theta mismatch"
        )

    @_skip_no_sklearn
    def test_log_prior_matches_sklearn(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_naive_bayes
        clf, _, _ = fitted_model
        stage = _parse_naive_bayes(clf)
        np.testing.assert_allclose(
            np.array(stage.log_prior), np.log(clf.class_prior_), rtol=1e-5,
        )

    @_skip_no_sklearn
    def test_log_var_const_formula(self, fitted_model):
        """log_var_const[c,f] == -0.5*log(2π*var[c,f])."""
        import math
        from timber.frontends.sklearn_parser import _parse_naive_bayes
        clf, _, _ = fitted_model
        stage = _parse_naive_bayes(clf)
        for c in range(stage.n_classes):
            for f in range(stage.n_features):
                expected = -0.5 * math.log(2.0 * math.pi * float(clf.var_[c, f]))
                assert abs(stage.log_var_const[c][f] - expected) < 1e-8

    @_skip_no_sklearn
    def test_inv_2var_formula(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_naive_bayes
        clf, _, _ = fitted_model
        stage = _parse_naive_bayes(clf)
        for c in range(stage.n_classes):
            for f in range(stage.n_features):
                expected = 1.0 / (2.0 * float(clf.var_[c, f]))
                assert abs(stage.inv_2var[c][f] - expected) < 1e-8

    @_skip_no_sklearn
    def test_convert_sklearn_dispatch(self, fitted_model):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import NaiveBayesStage
        clf, _, _ = fitted_model
        ir = _convert_sklearn(clf)
        assert isinstance(ir.pipeline[0], NaiveBayesStage)
        assert ir.schema.n_outputs == 2


class TestNaiveBayesC99:
    @pytest.fixture
    def ir(self):
        from sklearn.naive_bayes import GaussianNB
        from timber.frontends.sklearn_parser import _parse_naive_bayes
        from timber.ir.model import TimberIR, Schema, Field, FieldType, Metadata
        rng = _rng()
        X = rng.normal(size=(300, 5))
        y = (X[:, 0] + X[:, 2] > 0).astype(int)
        clf = GaussianNB()
        clf.fit(X, y)
        stage = _parse_naive_bayes(clf)
        ir = TimberIR(
            pipeline=[stage],
            schema=Schema(
                input_fields=[Field(f"f{i}", FieldType.FLOAT32, i) for i in range(5)],
                output_fields=[Field(f"c{i}", FieldType.FLOAT32, i) for i in range(2)],
            ),
            metadata=Metadata(source_framework="sklearn"),
        )
        return ir, clf, X

    @_skip_no_sklearn
    def test_emit_contains_softmax(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        c = C99Emitter().emit(ir_obj).model_c
        assert "softmax" in c or "sm" in c

    @_skip_no_sklearn
    def test_header_n_outputs_2(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        h = C99Emitter().emit(ir_obj).model_h
        assert "TIMBER_N_OUTPUTS  2" in h

    @_skip_no_sklearn
    def test_data_contains_theta(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        dc = C99Emitter().emit(ir_obj).model_data_c
        assert "TIMBER_NB_THETA" in dc

    @_skip_no_sklearn
    def test_compiles(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _ = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        assert so.exists()

    @_skip_no_sklearn
    def test_probabilities_sum_to_1(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, clf, X = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        result = _ctypes_infer(so, X[:50].astype(np.float32), 2)
        assert result is not None
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    @_skip_no_sklearn
    def test_probabilities_match_sklearn(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, clf, X = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        X_test = X[:50].astype(np.float32)
        result = _ctypes_infer(so, X_test, 2)
        ref = clf.predict_proba(X_test)
        np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)

    @_skip_no_sklearn
    def test_argmax_matches_sklearn_predict(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, clf, X = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        X_test = X[:100].astype(np.float32)
        result = _ctypes_infer(so, X_test, 2)
        timber_pred = np.argmax(result, axis=1)
        sklearn_pred = clf.predict(X_test)
        acc = np.mean(timber_pred == sklearn_pred)
        assert acc >= 0.99, f"Argmax accuracy vs sklearn.predict: {acc:.2%}"


class TestNaiveBayesEndToEnd:
    @_skip_no_sklearn
    def test_multiclass_3_classes(self, tmp_path):
        from sklearn.naive_bayes import GaussianNB
        from timber.frontends.sklearn_parser import parse_sklearn_model
        from timber.codegen.c99 import C99Emitter
        rng = _rng()
        X = rng.normal(size=(300, 4))
        y = np.where(X[:, 0] < -0.5, 0, np.where(X[:, 0] > 0.5, 2, 1))
        clf = GaussianNB()
        clf.fit(X, y)
        pkl = tmp_path / "gnb.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(clf, f)
        ir = parse_sklearn_model(str(pkl))
        build = tmp_path / "build"
        C99Emitter().emit(ir).write(build)
        so = _compile_so(build)
        if so is None:
            pytest.skip("no gcc")
        X_test = rng.normal(size=(30, 4)).astype(np.float32)
        result = _ctypes_infer(so, X_test, 3)
        assert result is not None
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-4)
        ref = clf.predict_proba(X_test)
        np.testing.assert_allclose(result, ref, atol=1e-4)


# ============================================================
# Gaussian Process Regressor
# ============================================================

class TestGPRIR:
    def _make_stage(self):
        from timber.ir.model import GPRStage
        return GPRStage(
            stage_name="test_gpr", stage_type="gpr",
            X_train=[[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
            alpha=[0.1, -0.2, 0.15],
            length_scale=1.0,
            amplitude=1.0,
            y_train_mean=0.5,
            y_train_std=1.0,
            n_features=2,
        )

    def test_stage_type(self):
        assert self._make_stage().stage_type == "gpr"

    def test_n_train(self):
        assert self._make_stage().n_train == 3

    def test_serialization_round_trip(self):
        from timber.ir.model import TimberIR, GPRStage
        s = self._make_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        assert isinstance(s2, GPRStage)
        assert s2.n_train == 3
        assert abs(s2.length_scale - 1.0) < 1e-9
        assert abs(s2.y_train_mean - 0.5) < 1e-9

    def test_alpha_preserved(self):
        from timber.ir.model import TimberIR
        s = self._make_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        np.testing.assert_allclose(s2.alpha, [0.1, -0.2, 0.15], atol=1e-7)

    def test_X_train_preserved(self):
        from timber.ir.model import TimberIR
        s = self._make_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        assert abs(s2.X_train[1][0] - 1.0) < 1e-9


class TestGPRParser:
    @pytest.fixture
    def fitted_model(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        rng = _rng()
        X = rng.uniform(0, 5, size=(40, 2))
        y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=40)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gpr.fit(X, y)
        return gpr, X, y

    @_skip_no_sklearn
    def test_parse_returns_gpr_stage(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_gpr
        from timber.ir.model import GPRStage
        gpr, _, _ = fitted_model
        assert isinstance(_parse_gpr(gpr), GPRStage)

    @_skip_no_sklearn
    def test_parse_n_features(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_gpr
        gpr, X, _ = fitted_model
        assert _parse_gpr(gpr).n_features == X.shape[1]

    @_skip_no_sklearn
    def test_parse_n_train(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_gpr
        gpr, X, _ = fitted_model
        assert _parse_gpr(gpr).n_train == len(X)

    @_skip_no_sklearn
    def test_parse_alpha_shape(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_gpr
        gpr, X, _ = fitted_model
        stage = _parse_gpr(gpr)
        assert len(stage.alpha) == len(X)

    @_skip_no_sklearn
    def test_X_train_matches_sklearn(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_gpr
        gpr, X, _ = fitted_model
        stage = _parse_gpr(gpr)
        np.testing.assert_allclose(np.array(stage.X_train), gpr.X_train_, rtol=1e-6)

    @_skip_no_sklearn
    def test_length_scale_extracted(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_gpr
        gpr, _, _ = fitted_model
        stage = _parse_gpr(gpr)
        assert stage.length_scale > 0

    @_skip_no_sklearn
    def test_amplitude_extracted(self, fitted_model):
        from timber.frontends.sklearn_parser import _parse_gpr
        gpr, _, _ = fitted_model
        stage = _parse_gpr(gpr)
        assert stage.amplitude > 0

    @_skip_no_sklearn
    def test_rbf_only_kernel(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from timber.frontends.sklearn_parser import _parse_gpr
        rng = _rng()
        X = rng.uniform(0, 3, size=(20, 1))
        y = np.sin(X.ravel())
        gpr = GaussianProcessRegressor(kernel=RBF(1.0), random_state=0)
        gpr.fit(X, y)
        stage = _parse_gpr(gpr)
        assert abs(stage.amplitude - 1.0) < 1e-6  # ConstantKernel amplitude is 1

    @_skip_no_sklearn
    def test_convert_sklearn_dispatch(self, fitted_model):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import GPRStage
        gpr, _, _ = fitted_model
        ir = _convert_sklearn(gpr)
        assert isinstance(ir.pipeline[0], GPRStage)
        assert ir.schema.n_outputs == 1


class TestGPRC99:
    @pytest.fixture
    def ir(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        from timber.frontends.sklearn_parser import _parse_gpr
        from timber.ir.model import TimberIR, Schema, Field, FieldType, Metadata
        rng = _rng()
        X = rng.uniform(0, 4, size=(30, 2))
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        kernel = ConstantKernel(1.5, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=1, normalize_y=True)
        gpr.fit(X, y)
        stage = _parse_gpr(gpr)
        ir = TimberIR(
            pipeline=[stage],
            schema=Schema(
                input_fields=[Field(f"f{i}", FieldType.FLOAT32, i) for i in range(2)],
                output_fields=[Field("y", FieldType.FLOAT32, 0)],
            ),
            metadata=Metadata(source_framework="sklearn"),
        )
        return ir, gpr, X, y

    @_skip_no_sklearn
    def test_emit_contains_exp(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _, _ = ir
        c = C99Emitter().emit(ir_obj).model_c
        assert "exp(" in c

    @_skip_no_sklearn
    def test_data_contains_gpr_arrays(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _, _ = ir
        dc = C99Emitter().emit(ir_obj).model_data_c
        assert "TIMBER_GPR_X" in dc
        assert "TIMBER_GPR_ALPHA" in dc
        assert "TIMBER_GPR_INV2LS2" in dc

    @_skip_no_sklearn
    def test_header_n_train(self, ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, gpr, X, _ = ir
        h = C99Emitter().emit(ir_obj).model_h
        assert f"TIMBER_N_TRAIN {len(X)}" in h

    @_skip_no_sklearn
    def test_compiles(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _, _ = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        assert so.exists()

    @_skip_no_sklearn
    def test_predictions_match_sklearn(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, gpr, X, _ = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        X_test = X[:20].astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        assert result is not None
        ref = gpr.predict(X_test).flatten()
        # Float32 storage for alpha and X_train → ~1e-4 abs tolerance
        np.testing.assert_allclose(result.flatten(), ref, atol=1e-3, rtol=1e-3)

    @_skip_no_sklearn
    def test_predictions_finite_and_bounded(self, ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, gpr, X, y = ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        X_test = X[:30].astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        assert np.all(np.isfinite(result))
        assert np.all(np.abs(result) < 10.0)


class TestGPREndToEnd:
    @_skip_no_sklearn
    def test_pkl_parse_and_compile(self, tmp_path):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from timber.frontends.sklearn_parser import parse_sklearn_model
        from timber.codegen.c99 import C99Emitter
        rng = _rng()
        X = rng.uniform(-2, 2, size=(25, 3))
        y = X[:, 0] ** 2 + X[:, 1]
        gpr = GaussianProcessRegressor(kernel=RBF(1.0), random_state=5)
        gpr.fit(X, y)
        pkl = tmp_path / "gpr.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(gpr, f)
        ir = parse_sklearn_model(str(pkl))
        build = tmp_path / "build"
        C99Emitter().emit(ir).write(build)
        so = _compile_so(build)
        if so is None:
            pytest.skip("no gcc")
        X_test = rng.uniform(-2, 2, size=(10, 3)).astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        assert result is not None
        ref = gpr.predict(X_test)
        np.testing.assert_allclose(result.flatten(), ref, atol=1e-3, rtol=1e-3)


# ============================================================
# k-NN
# ============================================================

class TestKNNIR:
    def _make_clf_stage(self):
        from timber.ir.model import KNNStage
        return KNNStage(
            stage_name="test_knn_clf", stage_type="knn",
            X_train=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            y_train=[[0.0], [0.0], [1.0], [1.0]],
            k=3,
            metric="euclidean",
            task_type="classifier",
            n_classes=2,
            n_features=2,
            n_outputs=1,
        )

    def _make_reg_stage(self):
        from timber.ir.model import KNNStage
        return KNNStage(
            stage_name="test_knn_reg", stage_type="knn",
            X_train=[[0.0], [1.0], [2.0], [3.0]],
            y_train=[[0.0], [1.0], [4.0], [9.0]],
            k=2,
            metric="euclidean",
            task_type="regressor",
            n_classes=0,
            n_features=1,
            n_outputs=1,
        )

    def test_stage_type_clf(self):
        assert self._make_clf_stage().stage_type == "knn"

    def test_n_train(self):
        assert self._make_clf_stage().n_train == 4

    def test_serialization_clf_round_trip(self):
        from timber.ir.model import TimberIR, KNNStage
        s = self._make_clf_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        assert isinstance(s2, KNNStage)
        assert s2.task_type == "classifier"
        assert s2.k == 3
        assert s2.n_classes == 2

    def test_serialization_reg_round_trip(self):
        from timber.ir.model import TimberIR, KNNStage
        s = self._make_reg_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        assert isinstance(s2, KNNStage)
        assert s2.task_type == "regressor"
        assert s2.k == 2

    def test_X_train_preserved(self):
        from timber.ir.model import TimberIR
        s = self._make_clf_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        s2 = ir2.pipeline[0]
        assert abs(s2.X_train[1][0] - 1.0) < 1e-9

    def test_metric_preserved(self):
        from timber.ir.model import TimberIR
        s = self._make_clf_stage()
        ir2 = TimberIR.from_dict(TimberIR(pipeline=[s]).to_dict())
        assert ir2.pipeline[0].metric == "euclidean"


class TestKNNParser:
    @pytest.fixture
    def clf_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        rng = _rng()
        X = rng.normal(size=(100, 4))
        y = (X[:, 0] > 0).astype(int)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        return clf, X, y

    @pytest.fixture
    def reg_model(self):
        from sklearn.neighbors import KNeighborsRegressor
        rng = _rng()
        X = rng.uniform(0, 3, size=(60, 2))
        y = X[:, 0] ** 2 + X[:, 1]
        reg = KNeighborsRegressor(n_neighbors=3)
        reg.fit(X, y)
        return reg, X, y

    @_skip_no_sklearn
    def test_clf_returns_knn_stage(self, clf_model):
        from timber.frontends.sklearn_parser import _parse_knn
        from timber.ir.model import KNNStage
        clf, _, _ = clf_model
        assert isinstance(_parse_knn(clf), KNNStage)

    @_skip_no_sklearn
    def test_clf_task_type(self, clf_model):
        from timber.frontends.sklearn_parser import _parse_knn
        clf, _, _ = clf_model
        assert _parse_knn(clf).task_type == "classifier"

    @_skip_no_sklearn
    def test_clf_n_features(self, clf_model):
        from timber.frontends.sklearn_parser import _parse_knn
        clf, _, _ = clf_model
        assert _parse_knn(clf).n_features == 4

    @_skip_no_sklearn
    def test_clf_k(self, clf_model):
        from timber.frontends.sklearn_parser import _parse_knn
        clf, _, _ = clf_model
        assert _parse_knn(clf).k == 5

    @_skip_no_sklearn
    def test_clf_n_train(self, clf_model):
        from timber.frontends.sklearn_parser import _parse_knn
        clf, X, _ = clf_model
        assert _parse_knn(clf).n_train == len(X)

    @_skip_no_sklearn
    def test_reg_task_type(self, reg_model):
        from timber.frontends.sklearn_parser import _parse_knn
        reg, _, _ = reg_model
        assert _parse_knn(reg).task_type == "regressor"

    @_skip_no_sklearn
    def test_clf_X_train_matches(self, clf_model):
        from timber.frontends.sklearn_parser import _parse_knn
        clf, X, _ = clf_model
        stage = _parse_knn(clf)
        np.testing.assert_allclose(np.array(stage.X_train), clf._fit_X, rtol=1e-6)

    @_skip_no_sklearn
    def test_convert_sklearn_clf_dispatch(self, clf_model):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import KNNStage
        clf, _, _ = clf_model
        ir = _convert_sklearn(clf)
        assert isinstance(ir.pipeline[0], KNNStage)

    @_skip_no_sklearn
    def test_convert_sklearn_reg_dispatch(self, reg_model):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import KNNStage
        reg, _, _ = reg_model
        ir = _convert_sklearn(reg)
        assert isinstance(ir.pipeline[0], KNNStage)


class TestKNNC99:
    @pytest.fixture
    def clf_ir(self):
        from sklearn.neighbors import KNeighborsClassifier
        from timber.frontends.sklearn_parser import _parse_knn
        from timber.ir.model import TimberIR, Schema, Field, FieldType, Metadata
        rng = _rng()
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        stage = _parse_knn(clf)
        ir = TimberIR(
            pipeline=[stage],
            schema=Schema(
                input_fields=[Field(f"f{i}", FieldType.FLOAT32, i) for i in range(3)],
                output_fields=[Field("cls", FieldType.FLOAT32, 0)],
            ),
            metadata=Metadata(source_framework="sklearn"),
        )
        return ir, clf, X, y

    @pytest.fixture
    def reg_ir(self):
        from sklearn.neighbors import KNeighborsRegressor
        from timber.frontends.sklearn_parser import _parse_knn
        from timber.ir.model import TimberIR, Schema, Field, FieldType, Metadata
        rng = _rng()
        X = rng.uniform(0, 2, size=(50, 2))
        y = X[:, 0] ** 2
        reg = KNeighborsRegressor(n_neighbors=3)
        reg.fit(X, y)
        stage = _parse_knn(reg)
        ir = TimberIR(
            pipeline=[stage],
            schema=Schema(
                input_fields=[Field(f"f{i}", FieldType.FLOAT32, i) for i in range(2)],
                output_fields=[Field("y", FieldType.FLOAT32, 0)],
            ),
            metadata=Metadata(source_framework="sklearn"),
        )
        return ir, reg, X, y

    @_skip_no_sklearn
    def test_clf_emit_contains_votes(self, clf_ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _, _ = clf_ir
        c = C99Emitter().emit(ir_obj).model_c
        assert "votes" in c

    @_skip_no_sklearn
    def test_reg_emit_contains_acc(self, reg_ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _, _ = reg_ir
        c = C99Emitter().emit(ir_obj).model_c
        assert "acc" in c

    @_skip_no_sklearn
    def test_data_contains_knn_x(self, clf_ir):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _, _ = clf_ir
        dc = C99Emitter().emit(ir_obj).model_data_c
        assert "TIMBER_KNN_X" in dc
        assert "TIMBER_KNN_Y" in dc

    @_skip_no_sklearn
    def test_clf_compiles(self, clf_ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _, _ = clf_ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        assert so.exists()

    @_skip_no_sklearn
    def test_reg_compiles(self, reg_ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, _, _, _ = reg_ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        assert so.exists()

    @_skip_no_sklearn
    def test_clf_predictions_match_sklearn(self, clf_ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, clf, X, _ = clf_ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        X_test = X[:40].astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        assert result is not None
        timber_pred = result.flatten().astype(int)
        sklearn_pred = clf.predict(X_test).astype(int)
        acc = np.mean(timber_pred == sklearn_pred)
        assert acc >= 0.95, f"KNN classifier accuracy: {acc:.2%}"

    @_skip_no_sklearn
    def test_reg_predictions_match_sklearn(self, reg_ir, tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir_obj, reg, X, _ = reg_ir
        C99Emitter().emit(ir_obj).write(tmp_path)
        so = _compile_so(tmp_path)
        if so is None:
            pytest.skip("no gcc")
        X_test = X[:25].astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        ref = reg.predict(X_test)
        np.testing.assert_allclose(result.flatten(), ref, atol=1e-4, rtol=1e-4)


class TestKNNEndToEnd:
    @_skip_no_sklearn
    def test_classifier_pkl(self, tmp_path):
        from sklearn.neighbors import KNeighborsClassifier
        from timber.frontends.sklearn_parser import parse_sklearn_model
        from timber.codegen.c99 import C99Emitter
        rng = _rng()
        X = rng.normal(size=(100, 4))
        y = (X[:, 0] > 0).astype(int)
        clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
        clf.fit(X, y)
        pkl = tmp_path / "knn_clf.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(clf, f)
        ir = parse_sklearn_model(str(pkl))
        build = tmp_path / "build"
        C99Emitter().emit(ir).write(build)
        so = _compile_so(build)
        if so is None:
            pytest.skip("no gcc")
        X_test = rng.normal(size=(20, 4)).astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        assert result is not None
        timber_pred = result.flatten().astype(int)
        sklearn_pred = clf.predict(X_test)
        assert np.mean(timber_pred == sklearn_pred) >= 0.95

    @_skip_no_sklearn
    def test_regressor_pkl(self, tmp_path):
        from sklearn.neighbors import KNeighborsRegressor
        from timber.frontends.sklearn_parser import parse_sklearn_model
        from timber.codegen.c99 import C99Emitter
        rng = _rng()
        X = rng.uniform(0, 3, size=(80, 2))
        y = np.sin(X[:, 0]) + X[:, 1]
        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)
        pkl = tmp_path / "knn_reg.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(reg, f)
        ir = parse_sklearn_model(str(pkl))
        build = tmp_path / "build"
        C99Emitter().emit(ir).write(build)
        so = _compile_so(build)
        if so is None:
            pytest.skip("no gcc")
        X_test = rng.uniform(0, 3, size=(15, 2)).astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        ref = reg.predict(X_test)
        np.testing.assert_allclose(result.flatten(), ref, atol=1e-4)

    @_skip_no_sklearn
    def test_manhattan_metric(self, tmp_path):
        from sklearn.neighbors import KNeighborsClassifier
        from timber.frontends.sklearn_parser import parse_sklearn_model
        from timber.codegen.c99 import C99Emitter
        rng = _rng()
        X = rng.normal(size=(80, 3))
        y = (X[:, 2] > 0).astype(int)
        clf = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
        clf.fit(X, y)
        pkl = tmp_path / "knn_manhattan.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(clf, f)
        ir = parse_sklearn_model(str(pkl))
        assert ir.pipeline[0].metric == "manhattan"
        build = tmp_path / "build"
        C99Emitter().emit(ir).write(build)
        so = _compile_so(build)
        if so is None:
            pytest.skip("no gcc")
        X_test = rng.normal(size=(15, 3)).astype(np.float32)
        result = _ctypes_infer(so, X_test, 1)
        assert result is not None
        assert np.mean(result.flatten().astype(int) == clf.predict(X_test)) >= 0.95


# ============================================================
# Cross-primitive: IR stage_type dispatch sanity
# ============================================================

class TestStageTypeDispatch:
    """Verify that all new stage types are correctly dispatched by _stage_from_dict."""

    @_skip_no_sklearn
    def test_iforest_stage_round_trip_type(self):
        from sklearn.ensemble import IsolationForest
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import IsolationForestStage, TimberIR
        rng = _rng()
        clf = IsolationForest(n_estimators=5, max_samples=32, random_state=0)
        clf.fit(rng.normal(size=(50, 3)))
        ir = _convert_sklearn(clf)
        ir2 = TimberIR.from_dict(ir.to_dict())
        assert isinstance(ir2.pipeline[0], IsolationForestStage)

    @_skip_no_sklearn
    def test_ocsvm_stage_round_trip_type(self):
        from sklearn.svm import OneClassSVM
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import SVMStage, TimberIR
        rng = _rng()
        clf = OneClassSVM(kernel="rbf", gamma=0.1)
        clf.fit(rng.normal(size=(50, 3)))
        ir = _convert_sklearn(clf)
        ir2 = TimberIR.from_dict(ir.to_dict())
        s = ir2.pipeline[0]
        assert isinstance(s, SVMStage)
        assert s.is_one_class is True

    @_skip_no_sklearn
    def test_nb_stage_round_trip_type(self):
        from sklearn.naive_bayes import GaussianNB
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import NaiveBayesStage, TimberIR
        rng = _rng()
        X = rng.normal(size=(50, 3))
        y = (X[:, 0] > 0).astype(int)
        clf = GaussianNB(); clf.fit(X, y)
        ir = _convert_sklearn(clf)
        ir2 = TimberIR.from_dict(ir.to_dict())
        assert isinstance(ir2.pipeline[0], NaiveBayesStage)

    @_skip_no_sklearn
    def test_gpr_stage_round_trip_type(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import GPRStage, TimberIR
        rng = _rng()
        X = rng.uniform(0, 2, size=(20, 2))
        y = X[:, 0]
        gpr = GaussianProcessRegressor(kernel=RBF(1.0), random_state=0)
        gpr.fit(X, y)
        ir = _convert_sklearn(gpr)
        ir2 = TimberIR.from_dict(ir.to_dict())
        assert isinstance(ir2.pipeline[0], GPRStage)

    @_skip_no_sklearn
    def test_knn_stage_round_trip_type(self):
        from sklearn.neighbors import KNeighborsClassifier
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.ir.model import KNNStage, TimberIR
        rng = _rng()
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = KNeighborsClassifier(n_neighbors=3); clf.fit(X, y)
        ir = _convert_sklearn(clf)
        ir2 = TimberIR.from_dict(ir.to_dict())
        assert isinstance(ir2.pipeline[0], KNNStage)

    @_skip_no_sklearn
    def test_all_five_c99_emit_no_exception(self, tmp_path):
        """Each primitive must emit without raising."""
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.naive_bayes import GaussianNB
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.neighbors import KNeighborsClassifier
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.codegen.c99 import C99Emitter
        rng = _rng()
        X = rng.normal(size=(50, 3))
        y = (X[:, 0] > 0).astype(int)

        # unsupervised: fit(X) only
        m_iforest = IsolationForest(n_estimators=5, max_samples=32, random_state=0)
        m_iforest.fit(X)
        m_ocsvm = OneClassSVM(kernel="rbf", gamma=0.5)
        m_ocsvm.fit(X)

        # supervised: fit(X, y)
        m_nb = GaussianNB()
        m_nb.fit(X, y)
        m_gpr = GaussianProcessRegressor(kernel=RBF(1.0), random_state=0)
        m_gpr.fit(X, y.astype(float))
        m_knn = KNeighborsClassifier(n_neighbors=3)
        m_knn.fit(X, y)

        for name, m in [
            ("IsolationForest", m_iforest),
            ("OneClassSVM",     m_ocsvm),
            ("GaussianNB",      m_nb),
            ("GPR",             m_gpr),
            ("KNN",             m_knn),
        ]:
            ir = _convert_sklearn(m)
            out = C99Emitter().emit(ir)
            assert out.model_c, f"{name} produced empty model.c"
            assert out.model_h, f"{name} produced empty model.h"
            assert out.model_data_c, f"{name} produced empty model_data.c"
