"""Tests for the AVX2 SIMD emitter."""

import pytest

from timber.codegen.c99 import TargetSpec
from timber.accel.accel.simd.avx2 import AVX2Emitter


@pytest.fixture
def avx2_emitter():
    target = TargetSpec(arch="x86_64", cpu_flags="-mavx2")
    return AVX2Emitter(target, simd_config={"unroll_factor": 2, "use_fma": True})


class TestAVX2EmitterMetadata:
    def test_instruction_set_name(self, avx2_emitter):
        assert avx2_emitter.instruction_set_name() == "avx2"

    def test_vector_width_bits(self, avx2_emitter):
        assert avx2_emitter.vector_width_bits() == 256

    def test_vector_width_constant(self, avx2_emitter):
        assert avx2_emitter.VECTOR_WIDTH == 8

    def test_compiler_flags_include_mavx2(self, avx2_emitter):
        flags = avx2_emitter.compiler_flags()
        assert "-mavx2" in flags

    def test_compiler_flags_include_fma(self, avx2_emitter):
        flags = avx2_emitter.compiler_flags()
        assert "-mfma" in flags

    def test_compiler_flags_no_fma_when_disabled(self):
        target = TargetSpec(arch="x86_64", cpu_flags="-mavx2")
        emitter = AVX2Emitter(target, simd_config={"use_fma": False})
        flags = emitter.compiler_flags()
        assert "-mfma" not in flags
        assert "-mavx2" in flags


class TestAVX2Includes:
    def test_immintrin_header(self, avx2_emitter):
        includes = avx2_emitter.emit_simd_includes()
        assert "#include <immintrin.h>" in includes

    def test_stdint_header(self, avx2_emitter):
        includes = avx2_emitter.emit_simd_includes()
        assert "#include <stdint.h>" in includes


class TestAVX2Traversal:
    def test_simple_ensemble_contains_mm256_intrinsics(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "_mm256_" in code

    def test_simple_ensemble_contains_load_ps(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "_mm256_load_ps" in code

    def test_simple_ensemble_contains_cmp_ps(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "_mm256_cmp_ps" in code

    def test_simple_ensemble_contains_movemask(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "_mm256_movemask_ps" in code

    def test_simple_ensemble_contains_add_ps(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "_mm256_add_ps" in code

    def test_simple_ensemble_contains_storeu(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "_mm256_storeu_ps" in code

    def test_simple_ensemble_has_scalar_fallback(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "_timber_avx2_traverse_single" in code
        assert "Scalar fallback for remainder" in code

    def test_multi_tree_ensemble_reports_correct_tree_count(self, avx2_emitter, multi_tree_ensemble):
        code = avx2_emitter.emit_simd_traversal(multi_tree_ensemble)
        assert "Trees: 2" in code

    def test_avx2_guard_ifdef(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "#ifdef __AVX2__" in code
        assert "#endif" in code

    def test_contains_entry_point(self, avx2_emitter, simple_ensemble):
        code = avx2_emitter.emit_simd_traversal(simple_ensemble)
        assert "timber_infer_simd" in code
