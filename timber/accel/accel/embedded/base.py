"""Base embedded emitter and factory — no-heap C code for microcontrollers."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass

from timber.codegen.c99 import C99Emitter, C99Output, TargetSpec
from timber.ir.model import TimberIR


@dataclass
class EmbeddedConstraints:
    """Hardware resource constraints from an AccelTargetProfile."""

    heap: bool = False
    stack_size: int = 4096
    flash_size: int = 256 * 1024
    ram_size: int = 64 * 1024


class EmbeddedEmitterBase(abc.ABC):
    """Abstract base for embedded-target code emitters.

    Wraps :class:`C99Emitter` output and applies microcontroller-specific
    transforms: static-only allocation, linker-section annotations, stack
    validation, and platform-specific headers/intrinsics.
    """

    def __init__(self, target: TargetSpec, embedded_config: dict):
        self.target = target
        self.embedded_config = embedded_config
        self._c99 = C99Emitter(target=target)
        self.constraints = EmbeddedConstraints(
            heap=embedded_config.get("heap", False),
            stack_size=embedded_config.get("stack_size", 4096),
            flash_size=embedded_config.get("flash_size", 256 * 1024),
            ram_size=embedded_config.get("ram_size", 64 * 1024),
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def platform_name(self) -> str:
        """Short identifier for the target platform."""
        ...

    @abc.abstractmethod
    def emit_platform_includes(self) -> str:
        """Return platform-specific ``#include`` directives."""
        ...

    @abc.abstractmethod
    def emit_startup_code(self, ir: TimberIR) -> str:
        """Generate platform startup / init code."""
        ...

    @abc.abstractmethod
    def emit_timing_code(self) -> str:
        """Generate inference-timing instrumentation."""
        ...

    @abc.abstractmethod
    def emit_linker_fragment(self, ir: TimberIR) -> str:
        """Generate a linker-script fragment for flash/ram layout."""
        ...

    # ------------------------------------------------------------------
    # Shared transforms
    # ------------------------------------------------------------------

    def emit(self, ir: TimberIR) -> C99Output:
        """Generate embedded-safe C code for *ir*."""
        baseline = self._c99.emit(ir)

        model_c = self._strip_dynamic_alloc(baseline.model_c)
        model_c = self._strip_stdio(model_c)
        model_c = self._add_volatile_qualifiers(model_c)
        model_c = self._inject_platform_code(model_c, ir)

        model_h = self._patch_header(baseline.model_h, ir)

        model_data_c = self._annotate_rodata(baseline.model_data_c)

        cmakelists = self._patch_cmake(baseline.cmakelists)
        makefile = self._patch_makefile(baseline.makefile)

        return C99Output(
            model_c=model_c,
            model_h=model_h,
            model_data_c=model_data_c,
            cmakelists=cmakelists,
            makefile=makefile,
        )

    # -- allocation removal --

    def _strip_dynamic_alloc(self, src: str) -> str:
        """Replace malloc/calloc/realloc/free with static buffer equivalents."""
        lines = src.split('\n')
        result = []
        for line in lines:
            if re.search(r'\b(malloc|calloc|realloc)\s*\(', line):
                result.append('/* EMBEDDED: dynamic allocation removed */')
            elif re.search(r'\bfree\s*\(', line):
                result.append('/* EMBEDDED: free removed */')
            else:
                result.append(line)
        return '\n'.join(result)

    def _strip_stdio(self, src: str) -> str:
        """Remove stdio includes (not available on bare-metal)."""
        return re.sub(r'#include\s*<stdio\.h>\s*\n?', '', src)

    def _add_volatile_qualifiers(self, src: str) -> str:
        """Add volatile to hardware-register pointers where needed."""
        # Mark peripheral base address casts as volatile
        src = re.sub(
            r'\(\s*(uint32_t|uint16_t|uint8_t)\s*\*\)',
            r'(volatile \1 *)',
            src,
        )
        return src

    def _annotate_rodata(self, data_c: str) -> str:
        """Add __attribute__((section(\".rodata\"))) to const data arrays."""
        data_c = re.sub(
            r'(static\s+const\s+(?:float|int|int32_t|uint32_t)\s+\w+\s*\[)',
            r'__attribute__((section(".rodata"))) \1',
            data_c,
        )
        return data_c

    def _inject_platform_code(self, model_c: str, ir: TimberIR) -> str:
        """Inject platform includes, startup, timing, and static buffers."""
        preamble_parts = [
            self.emit_platform_includes(),
            "",
            self._emit_static_buffers(ir),
            "",
            f"/* Stack budget: {self.constraints.stack_size} bytes */",
            f"/* Flash budget: {self.constraints.flash_size} bytes */",
            f"/* RAM   budget: {self.constraints.ram_size} bytes */",
            "",
            self.emit_timing_code(),
            "",
            self.emit_startup_code(ir),
            "",
        ]
        preamble = "\n".join(preamble_parts)

        # Insert after the last existing #include
        lines = model_c.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("#include"):
                insert_idx = i + 1
        lines.insert(insert_idx, "\n" + preamble)
        return "\n".join(lines)

    def _emit_static_buffers(self, ir: TimberIR) -> str:
        """Declare static inference buffers sized to model dimensions."""
        n_feat = ir.schema.get("n_features", 16) if hasattr(ir, "schema") and isinstance(ir.schema, dict) else 16
        n_out = ir.schema.get("n_outputs", 1) if hasattr(ir, "schema") and isinstance(ir.schema, dict) else 1
        lines = [
            "/* --- Static inference buffers (no heap) --- */",
            f"static float _timber_input_buf[{n_feat}];",
            f"static float _timber_output_buf[{n_out}];",
            f"static float _timber_scratch[{max(n_feat, n_out) * 2}];",
        ]
        return "\n".join(lines)

    def _patch_header(self, model_h: str, ir: TimberIR) -> str:
        """Add embedded defines and stack-usage comment to the header."""
        defines = [
            f"#define TIMBER_EMBEDDED 1",
            f"#define TIMBER_EMBEDDED_PLATFORM \"{self.platform_name()}\"",
            f"#define TIMBER_NO_HEAP 1",
            f"#define TIMBER_STACK_BUDGET {self.constraints.stack_size}",
        ]
        return "\n".join(defines) + "\n\n" + model_h

    def _patch_cmake(self, cmake: str) -> str:
        """Add embedded cross-compilation settings to CMakeLists."""
        prefix = self.target.cross_prefix
        flags = self.target.cpu_flags
        extra = (
            f"set(CMAKE_SYSTEM_NAME Generic)\n"
            f"set(CMAKE_SYSTEM_PROCESSOR {self.target.arch})\n"
            f"set(CMAKE_C_COMPILER {prefix}gcc)\n"
            f"set(CMAKE_C_FLAGS \"${{CMAKE_C_FLAGS}} {flags} -Os -ffunction-sections -fdata-sections\")\n"
            f"set(CMAKE_EXE_LINKER_FLAGS \"${{CMAKE_EXE_LINKER_FLAGS}} -Wl,--gc-sections\")\n"
        )
        return extra + "\n" + cmake

    def _patch_makefile(self, makefile: str) -> str:
        """Add embedded cross-compilation settings to Makefile."""
        prefix = self.target.cross_prefix
        flags = self.target.cpu_flags
        extra = (
            f"CC = {prefix}gcc\n"
            f"CFLAGS += {flags} -Os -ffunction-sections -fdata-sections\n"
            f"LDFLAGS += -Wl,--gc-sections\n"
        )
        return extra + "\n" + makefile


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedded_emitter(profile) -> EmbeddedEmitterBase:
    """Return the correct embedded emitter for an :class:`AccelTargetProfile`.

    Dispatches based on ``embedded_config["platform"]``:
    - ``"cortex_m"`` -> :class:`CortexMEmitter`
    - ``"esp32"``    -> :class:`ESP32Emitter`
    - ``"stm32"``    -> :class:`STM32Emitter`
    """
    from timber.accel.accel.embedded.cortex_m import CortexMEmitter
    from timber.accel.accel.embedded.esp32 import ESP32Emitter
    from timber.accel.accel.embedded.stm32 import STM32Emitter

    platform = profile.embedded_config.get("platform", "")
    mapping = {
        "cortex_m": CortexMEmitter,
        "esp32": ESP32Emitter,
        "stm32": STM32Emitter,
    }
    cls = mapping.get(platform)
    if cls is None:
        raise ValueError(
            f"Unknown embedded platform: {platform!r}. "
            f"Supported: {list(mapping)}"
        )
    return cls(target=profile.target_spec, embedded_config=profile.embedded_config)
