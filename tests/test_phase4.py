"""Tests for Phase 4 features:
  1. Broader ONNX operator support (Linear, SVM, Normalizer, Scaler)
  2. ARM Cortex-M / RISC-V embedded deployment profiles
  3. Enhanced MISRA-C compliance
  4. LLVM IR target backend
  5. Differential privacy inference
  6. Richer benchmark report
"""

from __future__ import annotations

import json
import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from timber.codegen.c99 import C99Emitter, TargetSpec, EMBEDDED_PROFILES
from timber.codegen.llvm_ir import LLVMIREmitter, LLVMIROutput, LLVM_TRIPLES
from timber.codegen.misra_c import MisraCEmitter, MisraReport
from timber.ir.model import (
    LinearStage,
    Metadata,
    NormalizerStage,
    Objective,
    ScalerStage,
    Schema,
    SVMStage,
    TimberIR,
    Tree,
    TreeEnsembleStage,
    TreeNode,
)
from timber.privacy.dp import (
    DPConfig,
    DPReport,
    apply_dp_noise,
    calibrate_epsilon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_ir(n_features: int = 4, n_trees: int = 2, n_classes: int = 1) -> TimberIR:
    """Minimal tree-ensemble IR for testing."""
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data.data[:100, :n_features].astype(np.float32), data.target[:100]
    m = xgb.XGBClassifier(n_estimators=n_trees, max_depth=2, random_state=42, eval_metric="logloss")
    m.fit(X, y)

    import tempfile, os
    from timber.frontends.xgboost_parser import parse_xgboost_json
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        m.get_booster().save_model(f.name)
        ir = parse_xgboost_json(f.name)
    os.unlink(f.name)
    return ir


def _make_linear_ir(n_features: int = 4, n_classes: int = 1) -> TimberIR:
    """Build a LinearStage-based IR directly."""
    if n_classes == 1:
        weights = [float(i + 1) * 0.1 for i in range(n_features)]
        stage = LinearStage(
            stage_name="linear",
            stage_type="linear",
            weights=weights,
            bias=0.5,
            activation="sigmoid",
            n_classes=1,
            multi_weights=False,
        )
    else:
        weights = [float(c * n_features + f) * 0.05 for c in range(n_classes) for f in range(n_features)]
        biases  = [float(c) * 0.1 for c in range(n_classes)]
        stage = LinearStage(
            stage_name="linear",
            stage_type="linear",
            weights=weights,
            bias=0.0,
            activation="softmax",
            n_classes=n_classes,
            multi_weights=True,
            biases=biases,
        )
    from timber.ir.model import Field, FieldType, Schema, Metadata
    schema = Schema(
        input_fields=[Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i) for i in range(n_features)],
        output_fields=[Field(name="out", dtype=FieldType.FLOAT32, index=0)],
    )
    return TimberIR(pipeline=[stage], schema=schema, metadata=Metadata())


def _make_svm_ir(n_features: int = 4, n_sv: int = 5) -> TimberIR:
    """Build an SVMStage-based IR directly."""
    rng = np.random.default_rng(42)
    sv_matrix = rng.standard_normal((n_sv, n_features)).tolist()
    dual_coef = rng.standard_normal(n_sv).tolist()
    stage = SVMStage(
        stage_name="svm",
        stage_type="svm",
        kernel_type="rbf",
        support_vectors=sv_matrix,
        dual_coef=dual_coef,
        rho=[0.1],
        n_support=[n_sv],
        gamma=0.5,
        coef0=0.0,
        degree=3,
        n_features=n_features,
        n_classes=2,
        objective=Objective.BINARY_CLASSIFICATION,
        post_transform="logistic",
    )
    from timber.ir.model import Field, FieldType, Schema, Metadata
    schema = Schema(
        input_fields=[Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i) for i in range(n_features)],
        output_fields=[Field(name="out", dtype=FieldType.FLOAT32, index=0)],
    )
    return TimberIR(pipeline=[stage], schema=schema, metadata=Metadata())


# ---------------------------------------------------------------------------
# 1. ONNX operator support — IR stage creation and C99 emission
# ---------------------------------------------------------------------------

class TestLinearStageC99:
    def test_binary_linear_header(self):
        ir = _make_linear_ir(n_features=4, n_classes=1)
        out = C99Emitter().emit(ir)
        assert "TIMBER_N_FEATURES" in out.model_h
        assert "TIMBER_MODEL_H" in out.model_h

    def test_binary_linear_data(self):
        ir = _make_linear_ir(n_features=4, n_classes=1)
        out = C99Emitter().emit(ir)
        assert "TIMBER_WEIGHTS" in out.model_data_c
        assert "TIMBER_BIAS" in out.model_data_c

    def test_binary_linear_inference_sigmoid(self):
        ir = _make_linear_ir(n_features=4, n_classes=1)
        out = C99Emitter().emit(ir)
        assert "exp" in out.model_c
        assert "1.0 / (1.0 + exp" in out.model_c

    def test_multiclass_linear_header(self):
        ir = _make_linear_ir(n_features=4, n_classes=3)
        out = C99Emitter().emit(ir)
        assert "TIMBER_N_CLASSES" in out.model_h

    def test_multiclass_linear_softmax(self):
        ir = _make_linear_ir(n_features=4, n_classes=3)
        out = C99Emitter().emit(ir)
        assert "softmax" in out.model_c
        assert "TIMBER_BIASES" in out.model_data_c

    def test_linear_batched_fn(self):
        ir = _make_linear_ir(n_features=4)
        out = C99Emitter().emit(ir)
        assert "timber_infer(" in out.model_c
        assert "n_samples" in out.model_c

    def test_linear_ir_serialization(self):
        ir = _make_linear_ir(n_features=4, n_classes=3)
        s = ir.to_json()
        ir2 = TimberIR.from_json(s)
        stage = ir2.pipeline[0]
        assert isinstance(stage, LinearStage)
        assert stage.n_classes == 3
        assert stage.multi_weights is True
        assert len(stage.weights) == 12


class TestSVMStageC99:
    def test_svm_header(self):
        ir = _make_svm_ir(n_features=4, n_sv=5)
        out = C99Emitter().emit(ir)
        assert "TIMBER_N_SV" in out.model_h
        assert "TIMBER_N_FEATURES" in out.model_h

    def test_svm_data_sv(self):
        ir = _make_svm_ir(n_features=4, n_sv=5)
        out = C99Emitter().emit(ir)
        assert "TIMBER_SV" in out.model_data_c
        assert "TIMBER_DUAL_COEF" in out.model_data_c
        assert "TIMBER_RHO" in out.model_data_c
        assert "TIMBER_GAMMA" in out.model_data_c

    def test_svm_rbf_kernel(self):
        ir = _make_svm_ir(n_features=4, n_sv=5)
        out = C99Emitter().emit(ir)
        assert "timber_kernel" in out.model_c
        assert "exp" in out.model_c
        assert "TIMBER_GAMMA" in out.model_c

    def test_svm_logistic_output(self):
        ir = _make_svm_ir(n_features=4, n_sv=5)
        out = C99Emitter().emit(ir)
        assert "1.0 / (1.0 + exp(" in out.model_c

    def test_svm_linear_kernel(self):
        ir = _make_svm_ir(n_features=3, n_sv=4)
        ir.pipeline[0].kernel_type = "linear"
        out = C99Emitter().emit(ir)
        assert "dot" in out.model_c

    def test_svm_poly_kernel(self):
        ir = _make_svm_ir(n_features=3, n_sv=4)
        ir.pipeline[0].kernel_type = "poly"
        out = C99Emitter().emit(ir)
        assert "TIMBER_DEGREE" in out.model_data_c

    def test_svm_ir_serialization(self):
        ir = _make_svm_ir(n_features=4, n_sv=5)
        s = ir.to_json()
        ir2 = TimberIR.from_json(s)
        stage = ir2.pipeline[0]
        assert isinstance(stage, SVMStage)
        assert stage.kernel_type == "rbf"
        assert len(stage.support_vectors) == 5


class TestNormalizerStage:
    def test_normalizer_ir_round_trip(self):
        from timber.ir.model import Field, FieldType, Schema, Metadata
        stage = NormalizerStage(stage_name="norm", stage_type="normalizer", norm="l2")
        schema = Schema(
            input_fields=[Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i) for i in range(4)],
            output_fields=[Field(name="out", dtype=FieldType.FLOAT32, index=0)],
        )
        ir = TimberIR(pipeline=[stage, _make_linear_ir(4).pipeline[0]], schema=schema, metadata=Metadata())
        s = ir.to_json()
        ir2 = TimberIR.from_json(s)
        ns = ir2.pipeline[0]
        assert isinstance(ns, NormalizerStage)
        assert ns.norm == "l2"

    def test_normalizer_l1(self):
        from timber.ir.model import Field, FieldType, Schema, Metadata
        stage = NormalizerStage(stage_name="norm", stage_type="normalizer", norm="l1")
        assert stage.norm == "l1"
        assert stage.stage_type == "normalizer"

    def test_normalizer_max(self):
        from timber.ir.model import Field, FieldType, Schema, Metadata
        stage = NormalizerStage(stage_name="norm", stage_type="normalizer", norm="max")
        assert stage.norm == "max"


# ---------------------------------------------------------------------------
# 2. Embedded deployment profiles
# ---------------------------------------------------------------------------

class TestEmbeddedProfiles:
    def test_known_profiles_exist(self):
        for profile in ("cortex-m4", "cortex-m33", "rv32imf", "rv64gc"):
            assert profile in EMBEDDED_PROFILES

    def test_cortex_m4_target_spec(self):
        spec = TargetSpec.for_embedded("cortex-m4")
        assert spec.embedded is True
        assert "arm-none-eabi-" in spec.cross_prefix
        assert "cortex-m4" in spec.cpu_flags
        assert "hard" in spec.cpu_flags

    def test_cortex_m33_target_spec(self):
        spec = TargetSpec.for_embedded("cortex-m33")
        assert spec.embedded is True
        assert "cortex-m33" in spec.cpu_flags
        assert "arm-none-eabi-" in spec.cross_prefix

    def test_rv32imf_target_spec(self):
        spec = TargetSpec.for_embedded("rv32imf")
        assert spec.embedded is True
        assert "riscv32" in spec.cross_prefix
        assert "rv32imf" in spec.cpu_flags

    def test_rv64gc_target_spec(self):
        spec = TargetSpec.for_embedded("rv64gc")
        assert spec.embedded is True
        assert "riscv64" in spec.cross_prefix
        assert "rv64gc" in spec.cpu_flags

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown embedded profile"):
            TargetSpec.for_embedded("arm-cortex-m99")

    def test_embedded_makefile_no_shared_lib(self):
        ir = _make_linear_ir(n_features=4)
        spec = TargetSpec.for_embedded("cortex-m4")
        out = C99Emitter(target=spec).emit(ir)
        assert "libtimber_model.so" not in out.makefile
        assert "libtimber_model.a" in out.makefile
        assert "arm-none-eabi-gcc" in out.makefile

    def test_embedded_makefile_no_fpic(self):
        ir = _make_linear_ir(n_features=4)
        spec = TargetSpec.for_embedded("rv32imf")
        out = C99Emitter(target=spec).emit(ir)
        assert "-fPIC" not in out.makefile

    def test_embedded_makefile_cpu_flags(self):
        ir = _make_svm_ir(n_features=4, n_sv=3)
        spec = TargetSpec.for_embedded("cortex-m4")
        out = C99Emitter(target=spec).emit(ir)
        assert "-mcpu=cortex-m4" in out.makefile

    def test_host_makefile_has_shared_lib(self):
        ir = _make_linear_ir(n_features=4)
        out = C99Emitter().emit(ir)
        assert "libtimber_model.so" in out.makefile

    def test_embedded_target_extra_flags(self):
        spec = TargetSpec.for_embedded("cortex-m4")
        assert "--specs=nosys.specs" in spec.extra_flags


# ---------------------------------------------------------------------------
# 3. Enhanced MISRA-C compliance
# ---------------------------------------------------------------------------

class TestMisraCEnhanced:
    def _make_ir(self, tmp_path: Path) -> TimberIR:
        return _make_simple_ir(n_features=10, n_trees=3)

    def test_emit_has_misra_banner(self, tmp_path):
        ir = self._make_ir(tmp_path)
        out = MisraCEmitter().emit(ir)
        assert "MISRA C:2012" in out.model_h
        assert "MISRA C:2012" in out.model_c

    def test_rule_1_1_compiler_extension(self):
        emitter = MisraCEmitter()
        code = '__attribute__((unused)) int x = 0;'
        report = emitter.check_compliance(code)
        assert not report.is_compliant
        rules = [v["rule"] for v in report.violations]
        assert "1.1" in rules

    def test_rule_7_1_octal_constant(self):
        emitter = MisraCEmitter()
        code = 'int x = 017;'
        report = emitter.check_compliance(code)
        rules = [v["rule"] for v in report.violations]
        assert "7.1" in rules

    def test_rule_20_9_stdio_include(self):
        emitter = MisraCEmitter()
        code = '#include <stdio.h>\nint x = 0;'
        report = emitter.check_compliance(code)
        assert not report.is_compliant
        rules = [v["rule"] for v in report.violations]
        assert "20.9" in rules

    def test_rule_21_1_null_redef(self):
        emitter = MisraCEmitter()
        code = '#define NULL ((void*)0)\nint x = 0;'
        report = emitter.check_compliance(code)
        assert not report.is_compliant
        rules = [v["rule"] for v in report.violations]
        assert "21.1" in rules

    def test_rule_21_6_printf_forbidden(self):
        emitter = MisraCEmitter()
        code = 'void foo(void) { printf("hello"); }'
        report = emitter.check_compliance(code)
        assert not report.is_compliant
        rules = [v["rule"] for v in report.violations]
        assert "21.6" in rules

    def test_rule_20_4_keyword_shadow(self):
        emitter = MisraCEmitter()
        code = '#define if(x) do_if(x)'
        report = emitter.check_compliance(code)
        assert not report.is_compliant
        rules = [v["rule"] for v in report.violations]
        assert "20.4" in rules

    def test_clean_generated_code_is_compliant(self):
        ir = _make_linear_ir(n_features=4)
        emitter = MisraCEmitter()
        out = emitter.emit(ir)
        report = emitter.check_compliance(out.model_c)
        assert report.rules_checked > 0
        assert report.rules_passed > 0
        assert report.is_compliant

    def test_transform_removes_attribute(self):
        emitter = MisraCEmitter()
        ir = _make_linear_ir(n_features=4)
        out = emitter.emit(ir)
        assert "__attribute__" not in out.model_c

    def test_transform_hex_u_suffix(self):
        emitter = MisraCEmitter()
        raw = "#define FOO 0xFF\nint x = 0xAB;"
        result = emitter._transform_source(raw)
        assert "0xFFU" in result or "0xFF" not in result.replace("0xFFU", "")

    def test_report_summary_string(self):
        emitter = MisraCEmitter()
        code = "int x = 0;"
        report = emitter.check_compliance(code)
        summary = report.summary()
        assert "MISRA C:2012" in summary
        assert "Rules checked" in summary

    def test_misra_report_violation_objects(self):
        emitter = MisraCEmitter()
        code = '__attribute__((unused)) int x;'
        report = emitter.check_compliance(code)
        assert len(report.violation_objects) > 0
        v = report.violation_objects[0]
        assert v.rule == "1.1"
        assert v.severity == "required"
        assert v.line > 0


# ---------------------------------------------------------------------------
# 4. LLVM IR backend
# ---------------------------------------------------------------------------

class TestLLVMIREmitter:
    def test_known_triples(self):
        for profile in ("x86_64", "aarch64", "cortex-m4", "rv32imf"):
            assert profile in LLVM_TRIPLES

    def test_emit_linear_module_header(self):
        ir = _make_linear_ir(n_features=4)
        emitter = LLVMIREmitter("x86_64")
        out = emitter.emit(ir)
        assert '; ModuleID' in out.model_ll
        assert 'target triple' in out.model_ll

    def test_emit_linear_function_defined(self):
        ir = _make_linear_ir(n_features=4)
        emitter = LLVMIREmitter("x86_64")
        out = emitter.emit(ir)
        assert 'define' in out.model_ll
        assert 'timber_infer_single' in out.model_ll

    def test_emit_linear_weights_constant(self):
        ir = _make_linear_ir(n_features=4)
        emitter = LLVMIREmitter("x86_64")
        out = emitter.emit(ir)
        assert 'timber_weights' in out.model_ll

    def test_emit_svm_rbf(self):
        ir = _make_svm_ir(n_features=4, n_sv=3)
        emitter = LLVMIREmitter("x86_64")
        out = emitter.emit(ir)
        assert 'timber_sv' in out.model_ll
        assert 'timber_dual_coef' in out.model_ll
        assert 'llvm.exp' in out.model_ll

    def test_emit_tree_ensemble(self):
        ir = _make_simple_ir(n_features=4, n_trees=2)
        emitter = LLVMIREmitter("x86_64")
        out = emitter.emit(ir)
        assert 'traverse_tree_0' in out.model_ll
        assert 'traverse_tree_1' in out.model_ll
        assert 'fcmp olt' in out.model_ll

    def test_aarch64_triple(self):
        ir = _make_linear_ir(n_features=4)
        emitter = LLVMIREmitter("aarch64")
        out = emitter.emit(ir)
        assert "aarch64" in out.target_triple

    def test_cortex_m4_triple(self):
        ir = _make_linear_ir(n_features=4)
        emitter = LLVMIREmitter("cortex-m4")
        out = emitter.emit(ir)
        assert "thumb" in out.target_triple.lower() or "arm" in out.target_triple.lower()

    def test_save_to_disk(self, tmp_path):
        ir = _make_linear_ir(n_features=4)
        emitter = LLVMIREmitter("x86_64")
        out = emitter.emit(ir)
        files = out.save(tmp_path)
        assert "model.ll" in files
        assert Path(files["model.ll"]).exists()
        content = Path(files["model.ll"]).read_text()
        assert 'timber_infer_single' in content

    def test_output_type(self):
        ir = _make_linear_ir(n_features=4)
        emitter = LLVMIREmitter("x86_64")
        out = emitter.emit(ir)
        assert isinstance(out, LLVMIROutput)
        assert isinstance(out.model_ll, str)
        assert len(out.model_ll) > 100

    def test_no_supported_stage_raises(self):
        from timber.ir.model import Field, FieldType, Schema, Metadata
        stage = NormalizerStage(stage_name="norm", stage_type="normalizer", norm="l2")
        schema = Schema(
            input_fields=[Field(name="f0", dtype=FieldType.FLOAT32, index=0)],
            output_fields=[Field(name="out", dtype=FieldType.FLOAT32, index=0)],
        )
        ir = TimberIR(pipeline=[stage], schema=schema, metadata=Metadata())
        with pytest.raises(ValueError, match="No supported primary stage"):
            LLVMIREmitter().emit(ir)


# ---------------------------------------------------------------------------
# 5. Differential privacy
# ---------------------------------------------------------------------------

class TestDifferentialPrivacy:
    def test_laplace_output_shape(self):
        outputs = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        cfg = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=42)
        noisy, report = apply_dp_noise(outputs, cfg)
        assert noisy.shape == outputs.shape

    def test_gaussian_output_shape(self):
        outputs = np.array([[0.5], [0.9]], dtype=np.float64)
        cfg = DPConfig(mechanism="gaussian", epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=42)
        noisy, report = apply_dp_noise(outputs, cfg)
        assert noisy.shape == outputs.shape

    def test_clipping_applied(self):
        outputs = np.ones((10, 1), dtype=np.float64) * 0.5
        cfg = DPConfig(mechanism="laplace", epsilon=0.01, sensitivity=1.0,
                       clip_outputs=True, output_min=0.0, output_max=1.0, seed=0)
        noisy, _ = apply_dp_noise(outputs, cfg)
        assert np.all(noisy >= 0.0)
        assert np.all(noisy <= 1.0)

    def test_no_clipping(self):
        outputs = np.ones((10, 1), dtype=np.float64) * 0.5
        cfg = DPConfig(mechanism="laplace", epsilon=0.001, sensitivity=10.0,
                       clip_outputs=False, seed=1)
        noisy, _ = apply_dp_noise(outputs, cfg)
        # With very small epsilon and large sensitivity, some values should exceed [0,1]
        # But at least outputs are different from inputs
        assert not np.allclose(noisy, outputs)

    def test_laplace_noise_scale(self):
        cfg = DPConfig(mechanism="laplace", epsilon=2.0, sensitivity=1.0)
        assert math.isclose(cfg.laplace_scale, 0.5, rel_tol=1e-9)

    def test_gaussian_sigma(self):
        cfg = DPConfig(mechanism="gaussian", epsilon=1.0, delta=1e-5, sensitivity=1.0)
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
        assert math.isclose(cfg.gaussian_sigma, expected, rel_tol=1e-6)

    def test_report_fields(self):
        outputs = np.array([[0.7]], dtype=np.float64)
        cfg = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=7)
        _, report = apply_dp_noise(outputs, cfg)
        assert isinstance(report, DPReport)
        assert report.mechanism == "laplace"
        assert report.epsilon == 1.0
        assert report.n_outputs_noised == 1
        assert report.actual_noise_l2 >= 0.0

    def test_report_summary(self):
        outputs = np.array([[0.5]], dtype=np.float64)
        cfg = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=0)
        _, report = apply_dp_noise(outputs, cfg)
        s = report.summary()
        assert "Laplace" in s or "laplace" in s
        assert "epsilon" in s or "ε" in s

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            DPConfig(epsilon=-1.0)

    def test_invalid_sensitivity_raises(self):
        with pytest.raises(ValueError, match="sensitivity"):
            DPConfig(sensitivity=0.0)

    def test_invalid_mechanism_raises(self):
        with pytest.raises(ValueError, match="mechanism"):
            DPConfig(mechanism="foo")

    def test_gaussian_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            DPConfig(mechanism="gaussian", delta=0.0)

    def test_deterministic_with_seed(self):
        outputs = np.array([[0.5, 0.3]], dtype=np.float64)
        cfg1 = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=123)
        cfg2 = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=123)
        noisy1, _ = apply_dp_noise(outputs, cfg1)
        noisy2, _ = apply_dp_noise(outputs, cfg2)
        np.testing.assert_array_equal(noisy1, noisy2)

    def test_different_seeds_differ(self):
        outputs = np.array([[0.5]], dtype=np.float64)
        cfg1 = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=1)
        cfg2 = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=2)
        noisy1, _ = apply_dp_noise(outputs, cfg1)
        noisy2, _ = apply_dp_noise(outputs, cfg2)
        assert not np.allclose(noisy1, noisy2)

    def test_high_epsilon_low_noise(self):
        """High ε = small noise, so noisy ≈ original."""
        rng = np.random.default_rng(0)
        outputs = rng.uniform(0.3, 0.7, (50, 1))
        cfg = DPConfig(mechanism="laplace", epsilon=1000.0, sensitivity=1.0, seed=0)
        noisy, _ = apply_dp_noise(outputs, cfg)
        assert np.mean(np.abs(noisy - outputs)) < 0.01

    def test_low_epsilon_high_noise(self):
        """Low ε = large noise."""
        outputs = np.full((100, 1), 0.5, dtype=np.float64)
        cfg = DPConfig(mechanism="laplace", epsilon=0.01, sensitivity=1.0,
                       clip_outputs=False, seed=0)
        noisy, report = apply_dp_noise(outputs, cfg)
        assert report.actual_noise_l2 > 1.0

    def test_calibrate_epsilon_laplace(self):
        eps = calibrate_epsilon(target_noise_std=0.1, sensitivity=1.0, mechanism="laplace")
        # std of Laplace(0, b) = b * sqrt(2); target=0.1 → b=0.1/sqrt(2) → eps=1/(b)=sqrt(2)/0.1
        expected = math.sqrt(2.0) / 0.1
        assert math.isclose(eps, expected, rel_tol=1e-6)

    def test_calibrate_epsilon_gaussian(self):
        eps = calibrate_epsilon(target_noise_std=0.5, sensitivity=1.0,
                                mechanism="gaussian", delta=1e-5)
        assert eps > 0.0

    def test_float32_output_preserved(self):
        outputs = np.array([[0.5]], dtype=np.float32)
        cfg = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0, seed=0)
        noisy, _ = apply_dp_noise(outputs, cfg)
        assert noisy.dtype == np.float32


# ---------------------------------------------------------------------------
# 6. Bench report generation
# ---------------------------------------------------------------------------

class TestBenchReport:
    def _make_report_data(self) -> dict:
        return {
            "timber_version": "0.2.0",
            "system": {
                "platform": "test-platform",
                "python": "3.11",
                "cpu": "test-cpu",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            "model": {
                "artifact": "test.json",
                "n_trees": 10,
                "max_depth": 3,
                "n_features": 30,
                "n_classes": 2,
                "objective": "binary:logistic",
                "n_samples": 100,
            },
            "results": [
                {
                    "batch_size": 1,
                    "n_runs": 200,
                    "min_us": 0.5,
                    "p50_us": 1.2,
                    "p95_us": 2.1,
                    "p99_us": 3.0,
                    "p999_us": 5.0,
                    "mean_us": 1.3,
                    "std_us": 0.4,
                    "cv_pct": 30.7,
                    "throughput_samples_per_sec": 833333.0,
                },
            ],
        }

    def test_json_report_valid(self, tmp_path):
        from timber.cli import _bench_report_html
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        report_path = tmp_path / "report.json"
        data = self._make_report_data()
        report_path.write_text(json.dumps(data, indent=2))
        loaded = json.loads(report_path.read_text())
        assert loaded["timber_version"] == "0.2.0"
        assert loaded["results"][0]["p99_us"] == 3.0

    def test_html_report_generation(self):
        from timber.cli import _bench_report_html
        data = self._make_report_data()
        html = _bench_report_html(data)
        assert "<!DOCTYPE html>" in html
        assert "Timber Benchmark Report" in html
        assert "P99" in html or "p99_us" in html
        assert "test-platform" in html
        assert "833333" in html

    def test_html_report_has_system_info(self):
        from timber.cli import _bench_report_html
        data = self._make_report_data()
        html = _bench_report_html(data)
        assert "test-cpu" in html
        assert "2024-01-01T00:00:00Z" in html

    def test_html_report_has_raw_json(self):
        from timber.cli import _bench_report_html
        data = self._make_report_data()
        html = _bench_report_html(data)
        assert "<pre>" in html
        assert '"timber_version"' in html


# ---------------------------------------------------------------------------
# 7. C99 dispatch on unsupported stage raises
# ---------------------------------------------------------------------------

class TestC99Dispatch:
    def test_empty_pipeline_raises(self):
        from timber.ir.model import Field, FieldType, Schema, Metadata
        schema = Schema(
            input_fields=[Field(name="f0", dtype=FieldType.FLOAT32, index=0)],
            output_fields=[Field(name="out", dtype=FieldType.FLOAT32, index=0)],
        )
        ir = TimberIR(pipeline=[], schema=schema, metadata=Metadata())
        with pytest.raises(ValueError, match="No supported primary stage"):
            C99Emitter().emit(ir)

    def test_normalizer_only_pipeline_raises(self):
        from timber.ir.model import Field, FieldType, Schema, Metadata
        stage = NormalizerStage(stage_name="norm", stage_type="normalizer", norm="l2")
        schema = Schema(
            input_fields=[Field(name="f0", dtype=FieldType.FLOAT32, index=0)],
            output_fields=[Field(name="out", dtype=FieldType.FLOAT32, index=0)],
        )
        ir = TimberIR(pipeline=[stage], schema=schema, metadata=Metadata())
        with pytest.raises(ValueError, match="No supported primary stage"):
            C99Emitter().emit(ir)
