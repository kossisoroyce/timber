"""Build helper for the generated native C++ server.

Invokes CMake configure + build to compile the server binary.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Outcome of a native server build."""

    success: bool
    binary_path: str | None = None
    build_dir: str | None = None
    stdout: str = ""
    stderr: str = ""


def build_server(
    project_dir: str | Path,
    *,
    build_type: str = "Release",
    cmake_extra: list[str] | None = None,
    jobs: int | None = None,
) -> BuildResult:
    """Configure and build the C++ server project.

    Parameters
    ----------
    project_dir:
        Root of the generated CMake project.
    build_type:
        CMake build type (Release, Debug, RelWithDebInfo).
    cmake_extra:
        Additional flags forwarded to ``cmake -S``.
    jobs:
        Parallel build jobs (defaults to system CPU count).

    Returns
    -------
    BuildResult
    """
    project = Path(project_dir).resolve()
    build_dir = project / "build"
    build_dir.mkdir(exist_ok=True)

    cmake = shutil.which("cmake")
    if cmake is None:
        return BuildResult(
            success=False, stderr="cmake not found on PATH"
        )

    # -- configure --
    configure_cmd = [
        cmake, "-S", str(project), "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    if cmake_extra:
        configure_cmd.extend(cmake_extra)

    logger.info("CMake configure: %s", " ".join(configure_cmd))
    cfg = subprocess.run(configure_cmd, capture_output=True, text=True)
    if cfg.returncode != 0:
        return BuildResult(
            success=False,
            build_dir=str(build_dir),
            stdout=cfg.stdout,
            stderr=cfg.stderr,
        )

    # -- build --
    build_cmd = [cmake, "--build", str(build_dir)]
    if jobs is not None:
        build_cmd += ["--parallel", str(jobs)]
    else:
        build_cmd += ["--parallel"]

    logger.info("CMake build: %s", " ".join(build_cmd))
    bld = subprocess.run(build_cmd, capture_output=True, text=True)

    binary = build_dir / "server"
    if not binary.exists():
        binary = None
    else:
        binary = str(binary)

    return BuildResult(
        success=bld.returncode == 0,
        binary_path=binary,
        build_dir=str(build_dir),
        stdout=bld.stdout,
        stderr=bld.stderr,
    )
