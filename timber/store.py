"""Timber model store — local registry and cache for compiled models.

Models are stored in ~/.timber/models/<name>/ with:
  - model_info.json  — metadata (name, format, size, loaded_at, etc.)
  - compiled/        — C99 artifacts (model.h, model.c, model_data.c, etc.)
  - lib/             — compiled shared library (.so/.dylib)

The registry is a JSON file at ~/.timber/registry.json mapping names to paths.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def _default_home() -> Path:
    return Path(os.environ.get("TIMBER_HOME", Path.home() / ".timber"))


@dataclass
class ModelInfo:
    """Metadata for a loaded model."""
    name: str
    source_path: str
    format: str
    n_trees: int = 0
    n_features: int = 0
    n_outputs: int = 0
    objective: str = ""
    framework: str = ""
    loaded_at: str = ""
    size_bytes: int = 0
    compiled: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ModelStore:
    """Manages the local model store."""

    def __init__(self, home: Path | None = None):
        self.home = home or _default_home()
        self.models_dir = self.home / "models"
        self.registry_path = self.home / "registry.json"
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.home.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _load_registry(self) -> dict[str, dict]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}

    def _save_registry(self, registry: dict[str, dict]):
        self.registry_path.write_text(json.dumps(registry, indent=2))

    def load_model(
        self,
        source_path: str | Path,
        name: str | None = None,
        format_hint: str | None = None,
    ) -> ModelInfo:
        """Compile and cache a model in the store.

        Args:
            source_path: Path to the model artifact file.
            name: Optional name for the model. Defaults to filename stem.
            format_hint: Optional format hint (xgboost, lightgbm, sklearn, etc.)

        Returns:
            ModelInfo for the loaded model.
        """
        from timber.frontends.auto_detect import detect_format, parse_model
        from timber.optimizer.pipeline import OptimizerPipeline
        from timber.codegen.c99 import C99Emitter

        source_path = Path(source_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {source_path}")

        if name is None:
            name = source_path.stem
        # Sanitize name
        name = name.replace(" ", "_").replace("/", "_").lower()

        fmt = format_hint or detect_format(str(source_path))
        if fmt is None:
            raise ValueError(f"Cannot detect format for '{source_path}'. Use --format.")

        # Parse
        ir = parse_model(str(source_path), format_hint=fmt)
        ensemble = ir.get_tree_ensemble()

        # Optimize
        opt_result = OptimizerPipeline().run(ir)
        ir = opt_result.ir
        ensemble = ir.get_tree_ensemble()

        # Create model directory
        model_dir = self.models_dir / name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)

        compiled_dir = model_dir / "compiled"
        lib_dir = model_dir / "lib"

        # Emit C99
        emitter = C99Emitter()
        output = emitter.emit(ir)
        output.write(compiled_dir)

        # Compile shared library
        lib_dir.mkdir(parents=True, exist_ok=True)
        lib_path = self._compile_shared_lib(compiled_dir, lib_dir)

        n_outputs = 1 if ensemble.n_classes <= 2 else ensemble.n_classes

        info = ModelInfo(
            name=name,
            source_path=str(source_path),
            format=fmt,
            n_trees=ensemble.n_trees,
            n_features=ensemble.n_features,
            n_outputs=n_outputs,
            objective=ensemble.objective.value,
            framework=ir.metadata.source_framework,
            loaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            size_bytes=sum(f.stat().st_size for f in compiled_dir.rglob("*") if f.is_file()),
            compiled=lib_path is not None,
        )

        # Save model info
        (model_dir / "model_info.json").write_text(json.dumps(info.to_dict(), indent=2))

        # Update registry
        registry = self._load_registry()
        registry[name] = info.to_dict()
        self._save_registry(registry)

        return info

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Look up a model by name."""
        registry = self._load_registry()
        if name not in registry:
            return None
        return ModelInfo.from_dict(registry[name])

    def list_models(self) -> list[ModelInfo]:
        """List all loaded models."""
        registry = self._load_registry()
        return [ModelInfo.from_dict(v) for v in registry.values()]

    def remove_model(self, name: str) -> bool:
        """Remove a model from the store."""
        registry = self._load_registry()
        if name not in registry:
            return False

        model_dir = self.models_dir / name
        if model_dir.exists():
            shutil.rmtree(model_dir)

        del registry[name]
        self._save_registry(registry)
        return True

    def get_model_dir(self, name: str) -> Optional[Path]:
        """Return the path to a model's directory."""
        d = self.models_dir / name
        return d if d.exists() else None

    def get_lib_path(self, name: str) -> Optional[Path]:
        """Return the path to a model's compiled shared library."""
        model_dir = self.models_dir / name
        lib_dir = model_dir / "lib"
        if not lib_dir.exists():
            return None
        for ext in (".dylib", ".so", ".dll"):
            lib = lib_dir / f"timber_model{ext}"
            if lib.exists():
                return lib
        return None

    def _compile_shared_lib(self, src_dir: Path, lib_dir: Path) -> Optional[Path]:
        """Compile C99 artifacts into a shared library."""
        import platform
        ext = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = lib_dir / f"timber_model{ext}"

        model_c = src_dir / "model.c"
        if not model_c.exists():
            return None

        cc = os.environ.get("CC", "gcc")
        flags = ["-shared", "-fPIC", "-O2", "-std=c99", "-lm"]
        if platform.system() == "Darwin":
            flags.append("-dynamiclib")

        try:
            subprocess.run(
                [cc] + flags + ["-o", str(lib_path), str(model_c)],
                capture_output=True, text=True, check=True,
                cwd=str(src_dir),
            )
            return lib_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
