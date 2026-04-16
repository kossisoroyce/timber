"""Base GPU emitter and factory."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from timber.codegen.c99 import C99Output, TargetSpec
from timber.ir.model import TimberIR


@dataclass
class GPUOutput:
    """GPU code generation output."""
    kernel_source: str       # .cu / .metal / .cl source
    host_source: str         # Host-side C code
    header: str              # Public header
    data_source: str         # Model data arrays
    build_script: str        # Build script (CMake or shell)

    def write(self, output_dir: str) -> list[str]:
        import os
        os.makedirs(output_dir, exist_ok=True)
        files = []
        for name, content in [
            ("kernel" + self._ext(), self.kernel_source),
            ("host.c", self.host_source),
            ("model.h", self.header),
            ("model_data.c", self.data_source),
            ("CMakeLists.txt", self.build_script),
        ]:
            path = os.path.join(output_dir, name)
            with open(path, "w") as f:
                f.write(content)
            files.append(path)
        return files

    def _ext(self) -> str:
        if "kernel" in self.kernel_source[:100].lower():
            if "__global__" in self.kernel_source:
                return ".cu"
            elif "kernel" in self.kernel_source and "metal" in self.kernel_source.lower():
                return ".metal"
        return ".cl"


class GPUEmitterBase(abc.ABC):
    """Abstract base for GPU code emitters."""

    def __init__(self, target: TargetSpec, gpu_config: dict):
        self.target = target
        self.gpu_config = gpu_config

    @abc.abstractmethod
    def backend_name(self) -> str: ...

    @abc.abstractmethod
    def emit_kernel(self, ir: TimberIR) -> str: ...

    @abc.abstractmethod
    def emit_host(self, ir: TimberIR) -> str: ...

    @abc.abstractmethod
    def emit_build_script(self, ir: TimberIR) -> str: ...

    def emit(self, ir: TimberIR) -> C99Output:
        """Generate GPU-accelerated code. Returns C99Output for CLI compatibility."""
        from timber.codegen.c99 import C99Emitter
        baseline = C99Emitter(target=self.target).emit(ir)

        kernel = self.emit_kernel(ir)
        host = self.emit_host(ir)
        build = self.emit_build_script(ir)

        # Pack into C99Output for CLI compatibility
        return C99Output(
            model_c=host + "\n\n/* GPU Kernel follows */\n" + kernel,
            model_h=baseline.model_h,
            model_data_c=baseline.model_data_c,
            cmakelists=build,
            makefile=baseline.makefile,
        )


def get_gpu_emitter(profile) -> GPUEmitterBase:
    from timber.accel.accel.gpu.cuda import CUDAEmitter
    from timber.accel.accel.gpu.metal import MetalEmitter
    from timber.accel.accel.gpu.opencl import OpenCLEmitter

    backend = profile.gpu_config.get("backend", "")
    mapping = {
        "cuda": CUDAEmitter,
        "metal": MetalEmitter,
        "opencl": OpenCLEmitter,
    }
    cls = mapping.get(backend)
    if cls is None:
        raise ValueError(f"Unknown GPU backend: {backend!r}. Supported: {list(mapping)}")
    return cls(target=profile.target_spec, gpu_config=profile.gpu_config)
