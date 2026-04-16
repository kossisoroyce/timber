"""Extended target profile loader for TimberAccelerate."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from timber.codegen.c99 import TargetSpec


@dataclass
class AccelTargetProfile:
    """Extended target profile with accelerator-specific configuration."""
    target_spec: TargetSpec
    name: str = ""
    simd_config: dict[str, Any] = field(default_factory=dict)
    gpu_config: dict[str, Any] = field(default_factory=dict)
    hls_config: dict[str, Any] = field(default_factory=dict)
    embedded_config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_simd(self) -> bool:
        return bool(self.simd_config)

    @property
    def is_gpu(self) -> bool:
        return bool(self.gpu_config)

    @property
    def is_hls(self) -> bool:
        return bool(self.hls_config)

    @property
    def is_embedded(self) -> bool:
        return bool(self.embedded_config)


# Built-in targets directory
_TARGETS_DIR = Path(__file__).resolve().parent.parent / "targets"


def load_target_profile(name_or_path: str) -> AccelTargetProfile:
    """Load a target profile from a TOML file or built-in name.

    Args:
        name_or_path: Either a path to a .toml file, or a built-in target name
                      (e.g., "x86_64_avx2_simd", "cuda_sm75", "embedded_cortex_m4").

    Returns:
        AccelTargetProfile with TargetSpec and accel-specific config.
    """
    path = Path(name_or_path)
    if not path.exists():
        # Try built-in targets
        builtin = _TARGETS_DIR / f"{name_or_path}.toml"
        if builtin.exists():
            path = builtin
        else:
            raise FileNotFoundError(
                f"Target profile not found: {name_or_path!r}. "
                f"Available built-in targets: {list_builtin_targets()}"
            )

    # Path traversal protection: ensure resolved path stays within the
    # expected targets directory when loading built-in targets.
    resolved = os.path.realpath(path)
    targets_base = os.path.realpath(_TARGETS_DIR)
    if not path.is_absolute():
        # For built-in target names, verify the resolved path is inside _TARGETS_DIR
        if not resolved.startswith(targets_base + os.sep) and resolved != targets_base:
            raise ValueError(
                f"Path traversal detected: resolved path {resolved!r} "
                f"escapes targets directory {targets_base!r}"
            )

    with open(resolved, "rb") as f:
        data = tomllib.load(f)

    # Normalize vector_width in [simd] section:
    # vector_width can be either an int (e.g. 128, 256) or the string "scalable"
    # (for SVE/RISC-V V targets). Numeric strings are converted to int.
    simd_section = data.get("simd", {})
    vw = simd_section.get("vector_width")
    if vw is not None and isinstance(vw, str) and vw != "scalable":
        try:
            simd_section["vector_width"] = int(vw)
        except ValueError:
            pass  # keep as-is if not a valid integer and not "scalable"

    target_data = data.get("target", {})
    target_spec = TargetSpec(
        arch=target_data.get("arch", "x86_64"),
        features=target_data.get("features", []),
        os=target_data.get("os", "linux"),
        abi=target_data.get("abi", "systemv"),
        precision=target_data.get("precision", "float32"),
        cross_prefix=target_data.get("cross_prefix", ""),
        cpu_flags=target_data.get("cpu_flags", ""),
        extra_flags=target_data.get("extra_flags", ""),
        embedded=target_data.get("embedded", False),
    )

    return AccelTargetProfile(
        target_spec=target_spec,
        name=target_data.get("name", path.stem),
        simd_config=data.get("simd", {}),
        gpu_config=data.get("gpu", {}),
        hls_config=data.get("hls", {}),
        embedded_config=data.get("embedded", {}),
    )


def list_builtin_targets() -> list[str]:
    """Return names of all built-in target profiles."""
    if not _TARGETS_DIR.exists():
        return []
    return sorted(p.stem for p in _TARGETS_DIR.glob("*.toml"))
