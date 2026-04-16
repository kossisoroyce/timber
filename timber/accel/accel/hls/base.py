"""Base HLS emitter and factory."""

from __future__ import annotations

import abc

from timber.codegen.c99 import C99Output, TargetSpec
from timber.ir.model import TimberIR


class HLSEmitterBase(abc.ABC):
    def __init__(self, target: TargetSpec, hls_config: dict):
        self.target = target
        self.hls_config = hls_config

    @abc.abstractmethod
    def vendor_name(self) -> str: ...

    @abc.abstractmethod
    def emit_hls_source(self, ir: TimberIR) -> str: ...

    @abc.abstractmethod
    def emit_testbench(self, ir: TimberIR) -> str: ...

    @abc.abstractmethod
    def emit_tcl_script(self, ir: TimberIR) -> str: ...

    def emit(self, ir: TimberIR) -> C99Output:
        from timber.codegen.c99 import C99Emitter
        baseline = C99Emitter(target=self.target).emit(ir)

        hls_src = self.emit_hls_source(ir)
        tb = self.emit_testbench(ir)
        tcl = self.emit_tcl_script(ir)

        return C99Output(
            model_c=hls_src,
            model_h=baseline.model_h + "\n/* HLS Top Function */\nvoid timber_infer_hls(const float* in, float* out, int n);\n",
            model_data_c=baseline.model_data_c,
            cmakelists=tcl,
            makefile=f"# Testbench\n{tb}",
        )


def get_hls_emitter(profile) -> HLSEmitterBase:
    from timber.accel.accel.hls.xilinx_vitis import XilinxVitisEmitter
    from timber.accel.accel.hls.intel_fpga import IntelFPGAEmitter

    vendor = profile.hls_config.get("vendor", "")
    mapping = {"xilinx": XilinxVitisEmitter, "intel": IntelFPGAEmitter}
    cls = mapping.get(vendor)
    if cls is None:
        raise ValueError(f"Unknown HLS vendor: {vendor!r}. Supported: {list(mapping)}")
    return cls(target=profile.target_spec, hls_config=profile.hls_config)
