"""Compilation smoke tests for generated C code.

These tests verify that the C code produced by the various emitters
actually compiles (or, for targets we cannot compile locally, passes
basic string-level sanity checks).
"""

import os
import platform
import shutil
import subprocess
import tempfile

import pytest

from timber.codegen.c99 import C99Emitter, TargetSpec
from timber.accel.accel.gpu.cuda import CUDAEmitter
from timber.accel.accel.simd.avx2 import AVX2Emitter
from timber.accel.accel.simd.avx512 import AVX512Emitter
from timber.accel.accel.simd.neon import NEONEmitter
from timber.accel.accel.simd.rvv import RVVEmitter
from timber.accel.accel.simd.sve import SVEEmitter

# ---------------------------------------------------------------------------
# Compiler detection (native + cross-compilers)
# ---------------------------------------------------------------------------

HAS_GCC = shutil.which("gcc") is not None or shutil.which("cc") is not None
CC = shutil.which("gcc") or shutil.which("cc") or "cc"

IS_X86 = platform.machine() in ("x86_64", "AMD64", "i386", "i686")
IS_ARM = platform.machine() in ("arm64", "aarch64", "armv7l")

# Cross-compiler detection — allows testing non-native targets on any host
_AARCH64_CROSS_COMPILERS = [
    "aarch64-elf-gcc",
    "aarch64-linux-gnu-gcc",
    "aarch64-none-elf-gcc",
    "aarch64-unknown-linux-gnu-gcc",
    "arm-linux-gnueabihf-gcc",
]
AARCH64_CC = next(
    (p for name in _AARCH64_CROSS_COMPILERS if (p := shutil.which(name))),
    None,
)

_RISCV64_CROSS_COMPILERS = [
    "riscv64-elf-gcc",
    "riscv64-linux-gnu-gcc",
    "riscv64-unknown-elf-gcc",
    "riscv64-unknown-linux-gnu-gcc",
]
RISCV64_CC = next(
    (p for name in _RISCV64_CROSS_COMPILERS if (p := shutil.which(name))),
    None,
)

HAS_NVCC = shutil.which("nvcc") is not None

# Can compile ARM targets if native ARM or cross-compiler available
CAN_COMPILE_ARM = IS_ARM or AARCH64_CC is not None
ARM_CC = CC if IS_ARM else AARCH64_CC

# Can compile RISC-V if cross-compiler available
CAN_COMPILE_RISCV = RISCV64_CC is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_c99_output(output, tmpdir: str) -> None:
    """Write all generated C files into *tmpdir*."""
    with open(os.path.join(tmpdir, "model.h"), "w") as fh:
        fh.write(output.model_h)
    with open(os.path.join(tmpdir, "model.c"), "w") as fh:
        fh.write(output.model_c)
    with open(os.path.join(tmpdir, "model_data.c"), "w") as fh:
        fh.write(output.model_data_c)


def _write_freestanding_stubs(tmpdir: str) -> None:
    """Write minimal stub headers for bare-metal cross-compilers.

    Bare-metal toolchains (``*-elf-gcc``) provide ``stdint.h`` and ``stddef.h``
    via ``-ffreestanding`` but lack hosted headers like ``string.h``, ``math.h``,
    ``stdio.h``, and ``stdlib.h``.  We supply stubs that declare only the
    symbols the generated C code actually uses.
    """
    stubs = {
        "string.h": (
            "#ifndef _STUB_STRING_H\n"
            "#define _STUB_STRING_H\n"
            "#include <stddef.h>\n"
            "void *memset(void *s, int c, size_t n);\n"
            "void *memcpy(void *dest, const void *src, size_t n);\n"
            "#endif\n"
        ),
        "math.h": (
            "#ifndef _STUB_MATH_H\n"
            "#define _STUB_MATH_H\n"
            "double fabs(double x);\n"
            "float fabsf(float x);\n"
            "double exp(double x);\n"
            "float expf(float x);\n"
            "double log(double x);\n"
            "int isnan(double x);\n"
            "#define NAN __builtin_nanf(\"\")\n"
            "#define INFINITY __builtin_inff()\n"
            "#endif\n"
        ),
        "stdio.h": (
            "#ifndef _STUB_STDIO_H\n"
            "#define _STUB_STDIO_H\n"
            "typedef struct FILE FILE;\n"
            "extern FILE *stderr;\n"
            "int fprintf(FILE *stream, const char *fmt, ...);\n"
            "#endif\n"
        ),
        "stdlib.h": (
            "#ifndef _STUB_STDLIB_H\n"
            "#define _STUB_STDLIB_H\n"
            "#include <stddef.h>\n"
            "void *malloc(size_t size);\n"
            "void free(void *ptr);\n"
            "#endif\n"
        ),
    }
    stub_dir = os.path.join(tmpdir, "stubs")
    os.makedirs(stub_dir, exist_ok=True)
    for name, content in stubs.items():
        with open(os.path.join(stub_dir, name), "w") as fh:
            fh.write(content)
    return stub_dir


def _compile_check(tmpdir: str, fname: str, extra_flags=None):
    """Run ``cc -fsyntax-only`` on a single file.  Returns the
    ``subprocess.CompletedProcess`` object.
    """
    cmd = [CC, "-fsyntax-only", "-std=c11", "-I", tmpdir]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append(os.path.join(tmpdir, fname))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def _assert_compiles(tmpdir: str, fname: str, extra_flags=None, label: str = ""):
    """Assert that *fname* passes a syntax-only compilation check."""
    result = _compile_check(tmpdir, fname, extra_flags=extra_flags)
    tag = f" ({label})" if label else ""
    assert result.returncode == 0, (
        f"{fname}{tag} failed to compile:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Baseline C99 tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GCC, reason="No C compiler available")
class TestBaselineC99Compilation:
    """Verify that baseline (non-SIMD) C99 output compiles."""

    def test_baseline_compiles_simple(self, simple_ensemble):
        target = TargetSpec(arch="x86_64")
        emitter = C99Emitter(target=target)
        output = emitter.emit(simple_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                _assert_compiles(tmpdir, fname, label="baseline/simple")

    def test_baseline_compiles_multi_tree(self, multi_tree_ensemble):
        target = TargetSpec(arch="x86_64")
        emitter = C99Emitter(target=target)
        output = emitter.emit(multi_tree_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                _assert_compiles(tmpdir, fname, label="baseline/multi_tree")

    def test_baseline_compiles_binary_classification(self, binary_classification_ensemble):
        target = TargetSpec(arch="x86_64")
        emitter = C99Emitter(target=target)
        output = emitter.emit(binary_classification_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                _assert_compiles(tmpdir, fname, label="baseline/binary_clf")


# ---------------------------------------------------------------------------
# AVX2 SIMD tests (x86_64 only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GCC or not IS_X86, reason="Needs x86 C compiler")
class TestAVX2Compilation:
    """Verify that AVX2-accelerated C output compiles on x86_64."""

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="x86_64", cpu_flags="-mavx2")
        return AVX2Emitter(target, simd_config={"unroll_factor": 2, "use_fma": True})

    def test_avx2_compiles_simple(self, simple_ensemble):
        emitter = self._make_emitter()
        output = emitter.emit(simple_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                _assert_compiles(
                    tmpdir, fname,
                    extra_flags=["-mavx2", "-mfma"],
                    label="avx2/simple",
                )

    def test_avx2_compiles_multi_tree(self, multi_tree_ensemble):
        emitter = self._make_emitter()
        output = emitter.emit(multi_tree_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                _assert_compiles(
                    tmpdir, fname,
                    extra_flags=["-mavx2", "-mfma"],
                    label="avx2/multi_tree",
                )


# ---------------------------------------------------------------------------
# AVX-512 SIMD tests (x86_64 only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GCC or not IS_X86, reason="Needs x86 C compiler")
class TestAVX512Compilation:
    """Verify that AVX-512-accelerated C output compiles on x86_64."""

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="x86_64", cpu_flags="-mavx512f")
        return AVX512Emitter(target, simd_config={"unroll_factor": 2, "use_fma": True})

    def test_avx512_compiles_simple(self, simple_ensemble):
        emitter = self._make_emitter()
        output = emitter.emit(simple_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                _assert_compiles(
                    tmpdir, fname,
                    extra_flags=["-mavx512f", "-mfma"],
                    label="avx512/simple",
                )

    def test_avx512_compiles_multi_tree(self, multi_tree_ensemble):
        emitter = self._make_emitter()
        output = emitter.emit(multi_tree_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                _assert_compiles(
                    tmpdir, fname,
                    extra_flags=["-mavx512f", "-mfma"],
                    label="avx512/multi_tree",
                )


# ---------------------------------------------------------------------------
# NEON SIMD tests (native ARM or aarch64 cross-compiler)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CAN_COMPILE_ARM,
    reason="Needs native ARM or aarch64 cross-compiler "
           "(e.g., aarch64-linux-gnu-gcc)",
)
class TestNEONCompilation:
    """Verify that NEON-accelerated C output compiles on ARM.

    Runs natively on ARM hosts, or via cross-compiler on x86_64
    (e.g., ``apt install gcc-aarch64-linux-gnu``).
    """

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="aarch64", cpu_flags="")
        return NEONEmitter(target, simd_config={"unroll_factor": 2})

    @staticmethod
    def _neon_compile_check(tmpdir: str, fname: str, label: str = ""):
        """Compile with the ARM compiler (native or cross)."""
        cmd = [ARM_CC, "-fsyntax-only", "-std=c11", "-I", tmpdir]
        if not IS_ARM:
            stub_dir = _write_freestanding_stubs(tmpdir)
            cmd.extend(["-march=armv8-a", "-ffreestanding",
                        "-isystem", stub_dir])
        cmd.append(os.path.join(tmpdir, fname))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        tag = f" ({label})" if label else ""
        assert result.returncode == 0, (
            f"{fname}{tag} failed to compile with {ARM_CC}:\n{result.stderr}"
        )

    def test_neon_compiles_simple(self, simple_ensemble):
        emitter = self._make_emitter()
        output = emitter.emit(simple_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                self._neon_compile_check(tmpdir, fname, label="neon/simple")

    def test_neon_compiles_multi_tree(self, multi_tree_ensemble):
        emitter = self._make_emitter()
        output = emitter.emit(multi_tree_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                self._neon_compile_check(tmpdir, fname, label="neon/multi_tree")


# ---------------------------------------------------------------------------
# SVE — cross-compilation when aarch64 toolchain available, else string checks
# ---------------------------------------------------------------------------

class TestSVEStringValidation:
    """String-level checks for SVE output (always run)."""

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="aarch64", cpu_flags="-march=armv8-a+sve")
        return SVEEmitter(target, simd_config={"unroll_factor": 2})

    def test_sve_no_python_isms(self, simple_ensemble):
        output = self._make_emitter().emit(simple_ensemble)
        for src_name, src in [("model.c", output.model_c), ("model_data.c", output.model_data_c)]:
            assert "None" not in src, f"Python 'None' leaked into {src_name}"
            assert " True" not in src, f"Python 'True' leaked into {src_name}"
            assert " False" not in src, f"Python 'False' leaked into {src_name}"

    def test_sve_balanced_braces(self, simple_ensemble):
        output = self._make_emitter().emit(simple_ensemble)
        for src_name, src in [("model.c", output.model_c), ("model_data.c", output.model_data_c)]:
            assert src.count("{") == src.count("}"), (
                f"Unbalanced braces in {src_name}: "
                f"{{ count={src.count('{')}, }} count={src.count('}')}"
            )


@pytest.mark.skipif(
    not CAN_COMPILE_ARM,
    reason="Needs aarch64 cross-compiler (e.g., aarch64-linux-gnu-gcc)",
)
class TestSVECompilation:
    """Verify SVE output compiles with an aarch64 toolchain that supports SVE."""

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="aarch64", cpu_flags="-march=armv8-a+sve")
        return SVEEmitter(target, simd_config={"unroll_factor": 2})

    @staticmethod
    def _sve_compile_check(tmpdir: str, fname: str, label: str = ""):
        stub_dir = _write_freestanding_stubs(tmpdir)
        cmd = [ARM_CC, "-fsyntax-only", "-std=c11", "-I", tmpdir,
               "-march=armv8-a+sve", "-ffreestanding", "-isystem", stub_dir]
        cmd.append(os.path.join(tmpdir, fname))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        tag = f" ({label})" if label else ""
        assert result.returncode == 0, (
            f"{fname}{tag} failed to compile with {ARM_CC}:\n{result.stderr}"
        )

    def test_sve_compiles_simple(self, simple_ensemble):
        output = self._make_emitter().emit(simple_ensemble)
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                self._sve_compile_check(tmpdir, fname, label="sve/simple")

    def test_sve_compiles_multi_tree(self, multi_tree_ensemble):
        output = self._make_emitter().emit(multi_tree_ensemble)
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                self._sve_compile_check(tmpdir, fname, label="sve/multi_tree")


# ---------------------------------------------------------------------------
# RVV — cross-compilation when riscv64 toolchain available, else string checks
# ---------------------------------------------------------------------------

class TestRVVStringValidation:
    """String-level checks for RISC-V V output (always run)."""

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="riscv64", cpu_flags="-march=rv64gcv")
        return RVVEmitter(target, simd_config={"unroll_factor": 2})

    def test_rvv_no_python_isms(self, simple_ensemble):
        output = self._make_emitter().emit(simple_ensemble)
        for src_name, src in [("model.c", output.model_c), ("model_data.c", output.model_data_c)]:
            assert "None" not in src, f"Python 'None' leaked into {src_name}"
            assert " True" not in src, f"Python 'True' leaked into {src_name}"
            assert " False" not in src, f"Python 'False' leaked into {src_name}"

    def test_rvv_balanced_braces(self, simple_ensemble):
        output = self._make_emitter().emit(simple_ensemble)
        for src_name, src in [("model.c", output.model_c), ("model_data.c", output.model_data_c)]:
            assert src.count("{") == src.count("}"), (
                f"Unbalanced braces in {src_name}: "
                f"{{ count={src.count('{')}, }} count={src.count('}')}"
            )


@pytest.mark.skipif(
    not CAN_COMPILE_RISCV,
    reason="Needs riscv64 cross-compiler (e.g., riscv64-linux-gnu-gcc)",
)
class TestRVVCompilation:
    """Verify RVV output compiles with a RISC-V toolchain that supports V extension."""

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="riscv64", cpu_flags="-march=rv64gcv")
        return RVVEmitter(target, simd_config={"unroll_factor": 2})

    @staticmethod
    def _rvv_compile_check(tmpdir: str, fname: str, label: str = ""):
        stub_dir = _write_freestanding_stubs(tmpdir)
        cmd = [RISCV64_CC, "-fsyntax-only", "-std=c11", "-I", tmpdir,
               "-march=rv64gcv", "-mabi=lp64d", "-ffreestanding",
               "-isystem", stub_dir]
        cmd.append(os.path.join(tmpdir, fname))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        tag = f" ({label})" if label else ""
        assert result.returncode == 0, (
            f"{fname}{tag} failed to compile with {RISCV64_CC}:\n{result.stderr}"
        )

    def test_rvv_compiles_simple(self, simple_ensemble):
        output = self._make_emitter().emit(simple_ensemble)
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                self._rvv_compile_check(tmpdir, fname, label="rvv/simple")

    def test_rvv_compiles_multi_tree(self, multi_tree_ensemble):
        output = self._make_emitter().emit(multi_tree_ensemble)
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_c99_output(output, tmpdir)
            for fname in ("model.c", "model_data.c"):
                self._rvv_compile_check(tmpdir, fname, label="rvv/multi_tree")


# ---------------------------------------------------------------------------
# CUDA — string-level validation (needs nvcc to actually compile)
# ---------------------------------------------------------------------------

class TestCUDAStringValidation:
    """String-level checks for CUDA output (always run)."""

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="x86_64", cpu_flags="")
        return CUDAEmitter(target, gpu_config={"block_size": 256, "use_shared_memory": False})

    def test_cuda_kernel_no_python_isms(self, simple_ensemble):
        emitter = self._make_emitter()
        kernel = emitter.emit_kernel(simple_ensemble)
        assert "None" not in kernel, "Python 'None' leaked into CUDA kernel"
        assert " True" not in kernel, "Python 'True' leaked into CUDA kernel"
        assert " False" not in kernel, "Python 'False' leaked into CUDA kernel"

    def test_cuda_kernel_balanced_braces(self, simple_ensemble):
        emitter = self._make_emitter()
        kernel = emitter.emit_kernel(simple_ensemble)
        assert kernel.count("{") == kernel.count("}"), (
            f"Unbalanced braces in CUDA kernel: "
            f"{{ count={kernel.count('{')}, }} count={kernel.count('}')}"
        )

    def test_cuda_host_balanced_braces(self, simple_ensemble):
        emitter = self._make_emitter()
        host = emitter.emit_host(simple_ensemble)
        assert host.count("{") == host.count("}"), (
            f"Unbalanced braces in CUDA host: "
            f"{{ count={host.count('{')}, }} count={host.count('}')}"
        )

    def test_cuda_multi_tree_no_python_isms(self, multi_tree_ensemble):
        emitter = self._make_emitter()
        kernel = emitter.emit_kernel(multi_tree_ensemble)
        host = emitter.emit_host(multi_tree_ensemble)
        for label, src in [("kernel", kernel), ("host", host)]:
            assert "None" not in src, f"Python 'None' leaked into CUDA {label}"
            assert " True" not in src, f"Python 'True' leaked into CUDA {label}"
            assert " False" not in src, f"Python 'False' leaked into CUDA {label}"


@pytest.mark.skipif(not HAS_NVCC, reason="Needs nvcc (CUDA toolkit)")
class TestCUDACompilation:
    """Verify CUDA output compiles with nvcc when available."""

    @staticmethod
    def _make_emitter():
        target = TargetSpec(arch="x86_64", cpu_flags="")
        return CUDAEmitter(target, gpu_config={"block_size": 256, "use_shared_memory": False})

    @staticmethod
    def _cuda_compile_check(tmpdir: str, fname: str, label: str = ""):
        cmd = ["nvcc", "--syntax-only", "-I", tmpdir,
               os.path.join(tmpdir, fname)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        tag = f" ({label})" if label else ""
        assert result.returncode == 0, (
            f"{fname}{tag} failed to compile with nvcc:\n{result.stderr}"
        )

    def test_cuda_kernel_compiles(self, simple_ensemble):
        emitter = self._make_emitter()
        kernel = emitter.emit_kernel(simple_ensemble)
        host = emitter.emit_host(simple_ensemble)

        with tempfile.TemporaryDirectory() as tmpdir:
            # CUDA needs model.h stub for the #include
            with open(os.path.join(tmpdir, "model.h"), "w") as f:
                f.write("#ifndef MODEL_H\n#define MODEL_H\n#endif\n")
            with open(os.path.join(tmpdir, "kernel.cu"), "w") as f:
                f.write(kernel)
            self._cuda_compile_check(tmpdir, "kernel.cu", label="cuda/kernel")
