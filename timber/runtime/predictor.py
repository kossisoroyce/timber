"""Drop-in Python predictor using compiled C shared library via ctypes."""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import numpy as np


class TimberPredictor:
    """Drop-in replacement for XGBoost/LightGBM predict() using compiled C inference.

    Usage:
        # From a compiled artifact directory
        predictor = TimberPredictor.from_artifact("./dist")
        predictions = predictor.predict(X)

        # From a model file (compiles on-the-fly)
        predictor = TimberPredictor.from_model("model.json")
        predictions = predictor.predict(X)

    The predictor automatically builds the shared library and exposes
    a numpy-compatible predict() interface identical to sklearn estimators.
    """

    def __init__(self, lib_path: str | Path, n_features: int, n_outputs: int, n_trees: int):
        self._lib_path = Path(lib_path)
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_trees = n_trees
        self._lib: Optional[ctypes.CDLL] = None
        self._ctx = None
        self._load_lib()

    def _load_lib(self) -> None:
        """Load the shared library and set up function signatures."""
        self._lib = ctypes.CDLL(str(self._lib_path))

        # timber_init
        self._lib.timber_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.timber_init.restype = ctypes.c_int

        # timber_free
        self._lib.timber_free.argtypes = [ctypes.c_void_p]
        self._lib.timber_free.restype = None

        # timber_infer
        self._lib.timber_infer.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # inputs
            ctypes.c_int,                     # n_samples
            ctypes.POINTER(ctypes.c_float),  # outputs
            ctypes.c_void_p,                  # ctx
        ]
        self._lib.timber_infer.restype = ctypes.c_int

        # timber_infer_single
        self._lib.timber_infer_single.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # inputs
            ctypes.POINTER(ctypes.c_float),  # outputs
            ctypes.c_void_p,                  # ctx
        ]
        self._lib.timber_infer_single.restype = ctypes.c_int

        # Initialize context
        ctx = ctypes.c_void_p()
        rc = self._lib.timber_init(ctypes.byref(ctx))
        if rc != 0:
            raise RuntimeError(f"timber_init failed with code {rc}")
        self._ctx = ctx

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on input array. Returns predictions.

        Args:
            X: Input features, shape (n_samples, n_features) or (n_features,).

        Returns:
            Predictions array, shape (n_samples,) or (n_samples, n_outputs).
        """
        if self._lib is None:
            raise RuntimeError("Library not loaded")

        X = np.ascontiguousarray(X, dtype=np.float32)
        single = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True

        n_samples = X.shape[0]
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )

        outputs = np.zeros((n_samples, self.n_outputs), dtype=np.float32)

        inputs_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        outputs_ptr = outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rc = self._lib.timber_infer(inputs_ptr, n_samples, outputs_ptr, self._ctx)
        if rc != 0:
            raise RuntimeError(f"timber_infer failed with code {rc}")

        if self.n_outputs == 1:
            outputs = outputs.ravel()

        if single:
            return outputs[0] if outputs.ndim == 1 else outputs[0]

        return outputs

    def predict_single(self, x: np.ndarray) -> float:
        """Run inference on a single sample. Returns scalar prediction."""
        x = np.ascontiguousarray(x, dtype=np.float32).ravel()
        if len(x) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {len(x)}"
            )

        output = np.zeros(self.n_outputs, dtype=np.float32)
        inputs_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        outputs_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rc = self._lib.timber_infer_single(inputs_ptr, outputs_ptr, self._ctx)
        if rc != 0:
            raise RuntimeError(f"timber_infer_single failed with code {rc}")

        return float(output[0])

    def close(self) -> None:
        """Free the model context."""
        if self._lib is not None and self._ctx is not None:
            self._lib.timber_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path, build: bool = True) -> "TimberPredictor":
        """Load a predictor from a compiled artifact directory.

        Args:
            artifact_dir: Path to the directory containing model.c, model.h, etc.
            build: If True, automatically build the shared library if not present.

        Returns:
            TimberPredictor instance ready for inference.
        """
        artifact_dir = Path(artifact_dir)

        # Determine shared lib name
        import sys
        if sys.platform == "darwin":
            lib_name = "libtimber_model.dylib"
        elif sys.platform == "win32":
            lib_name = "timber_model.dll"
        else:
            lib_name = "libtimber_model.so"

        lib_path = artifact_dir / lib_name

        if not lib_path.exists() and build:
            _build_shared_lib(artifact_dir, lib_path)

        if not lib_path.exists():
            raise FileNotFoundError(
                f"Shared library not found at {lib_path}. "
                f"Run 'make' in {artifact_dir} or pass build=True."
            )

        # Parse n_features and n_outputs from header
        header = (artifact_dir / "model.h").read_text()
        n_features = _parse_define(header, "TIMBER_N_FEATURES")
        n_outputs = _parse_define(header, "TIMBER_N_OUTPUTS")
        n_trees = _parse_define(header, "TIMBER_N_TREES")

        return cls(lib_path, n_features, n_outputs, n_trees)

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        format_hint: Optional[str] = None,
        optimize: bool = True,
        target: Optional[str] = None,
    ) -> "TimberPredictor":
        """Compile a model on-the-fly and return a ready predictor.

        Args:
            model_path: Path to the model file.
            format_hint: Model format (xgboost, lightgbm). Auto-detected if None.
            optimize: Whether to run optimizer passes.
            target: Path to a target spec TOML file.

        Returns:
            TimberPredictor instance ready for inference.
        """
        from timber.frontends import detect_format, parse_model
        from timber.optimizer.pipeline import OptimizerPipeline
        from timber.codegen.c99 import C99Emitter, TargetSpec

        model_path = Path(model_path)
        detected = format_hint or detect_format(str(model_path))
        if detected is None:
            raise ValueError(f"Cannot detect model format for '{model_path}'")

        ir = parse_model(str(model_path), format_hint=detected)

        if optimize:
            optimizer = OptimizerPipeline()
            result = optimizer.run(ir)
            ir = result.ir

        target_spec = TargetSpec()
        if target:
            import sys
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib
            with open(target, "rb") as f:
                data = tomllib.load(f)
            t = data.get("target", {})
            target_spec.arch = t.get("arch", "x86_64")
            target_spec.features = t.get("features", [])

        emitter = C99Emitter(target=target_spec)
        output = emitter.emit(ir)

        # Write to temp dir and build
        tmp_dir = Path(tempfile.mkdtemp(prefix="timber_"))
        output.write(tmp_dir)

        return cls.from_artifact(tmp_dir, build=True)


def _build_shared_lib(artifact_dir: Path, lib_path: Path) -> None:
    """Build the shared library from C source files."""
    import sys

    model_c = artifact_dir / "model.c"
    if not model_c.exists():
        raise FileNotFoundError(f"model.c not found in {artifact_dir}")

    if sys.platform == "darwin":
        shared_flag = "-dynamiclib"
    else:
        shared_flag = "-shared"

    cmd = [
        "gcc", "-std=c99", "-O3", "-Wall",
        shared_flag, "-fPIC",
        "-o", str(lib_path),
        str(model_c),
        "-lm",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(artifact_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build shared library:\n{result.stderr}\n{result.stdout}"
        )


def _parse_define(header: str, name: str) -> int:
    """Parse a #define integer value from a C header."""
    for line in header.splitlines():
        line = line.strip()
        if line.startswith(f"#define {name}"):
            parts = line.split()
            if len(parts) >= 3:
                return int(parts[2])
    raise ValueError(f"Could not find #define {name} in header")
