"""STM32 HAL embedded emitter — STM32CubeMX compatible, no-heap inference.

Generates C code targeting STM32 families (F4/F7/H7/L4) using:
- STM32 HAL headers for peripheral access
- DMA transfer hints for bulk data movement
- HAL timer-based inference timing
- STM32CubeMX project structure conventions
"""

from __future__ import annotations

import re

from timber.codegen.c99 import C99Output, TargetSpec
from timber.ir.model import TimberIR

from timber.accel.accel.embedded.base import EmbeddedEmitterBase


# Map architecture strings to STM32 HAL header include names
_HAL_HEADERS = {
    "stm32f4": "stm32f4xx_hal.h",
    "stm32f7": "stm32f7xx_hal.h",
    "stm32h7": "stm32h7xx_hal.h",
    "stm32l4": "stm32l4xx_hal.h",
    "stm32g4": "stm32g4xx_hal.h",
}


class STM32Emitter(EmbeddedEmitterBase):
    """Emit STM32 HAL compatible inference code."""

    def __init__(self, target: TargetSpec, embedded_config: dict):
        super().__init__(target, embedded_config)
        self._family = embedded_config.get("stm32_family", "stm32f7")
        self._hal_header = _HAL_HEADERS.get(self._family, "stm32f7xx_hal.h")
        self._use_dma = embedded_config.get("use_dma", False)
        self._timer_instance = embedded_config.get("timer_instance", "TIM2")

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def platform_name(self) -> str:
        return "stm32"

    def emit_platform_includes(self) -> str:
        lines = [
            '#include <stdint.h>',
            '#include <string.h>',
            f'#include "{self._hal_header}"',
        ]
        if self._use_dma:
            lines.append(f'#include "{self._family}xx_hal_dma.h"')
        return "\n".join(lines)

    def emit_startup_code(self, ir: TimberIR) -> str:
        """Generate HAL-based startup and inference wrapper."""
        lines = [
            "/* --- STM32 HAL Inference Wrapper --- */",
            "",
            "/* Forward-declare the HAL timer handle (defined by CubeMX) */",
            f"extern TIM_HandleTypeDef h{self._timer_instance.lower()};",
            "",
        ]

        if self._use_dma:
            lines += self._emit_dma_support()

        lines += [
            "/*",
            " * timber_embedded_init: one-time initialisation.",
            " * Call after HAL_Init() and SystemClock_Config().",
            " */",
            "static void timber_embedded_init(void) {",
            f"    /* Start timing timer ({self._timer_instance}) */",
            f"    HAL_TIM_Base_Start(&h{self._timer_instance.lower()});",
            "}",
            "",
            "/*",
            " * timber_embedded_infer: single-sample inference.",
            " * No heap, bounded stack, interrupt-safe.",
            " *",
            f" * Stack usage estimate: {self._estimate_stack(ir)} bytes",
            " */",
            "static int timber_embedded_infer(",
            "    const float* input,",
            "    float*       output,",
            "    const TimberCtx* ctx)",
            "{",
            "    /* Disable interrupts for deterministic timing */",
            "    uint32_t primask = __get_PRIMASK();",
            "    __disable_irq();",
            "",
            "    int rc = timber_infer(input, 1, output, ctx);",
            "",
            "    /* Restore interrupt state */",
            "    __set_PRIMASK(primask);",
            "    return rc;",
            "}",
        ]
        return "\n".join(lines)

    def emit_timing_code(self) -> str:
        """HAL timer-based inference timing."""
        tim = self._timer_instance
        htim = f"h{tim.lower()}"
        lines = [
            f"/* --- STM32 Inference Timing ({tim}) --- */",
            "",
            f"static volatile uint32_t _timber_timer_start;",
            "",
            "static inline void timber_timing_start(void) {",
            f"    __HAL_TIM_SET_COUNTER(&{htim}, 0);",
            f"    _timber_timer_start = __HAL_TIM_GET_COUNTER(&{htim});",
            "}",
            "",
            "static inline uint32_t timber_timing_stop(void) {",
            f"    uint32_t end = __HAL_TIM_GET_COUNTER(&{htim});",
            "    return end - _timber_timer_start;",
            "}",
        ]
        return "\n".join(lines)

    def emit_linker_fragment(self, ir: TimberIR) -> str:
        """STM32 linker script fragment for flash/ram layout."""
        flash = self.constraints.flash_size
        ram = self.constraints.ram_size
        lines = [
            "/* --- STM32 Linker Script Fragment --- */",
            "/*",
            " * Merge into your STM32CubeMX-generated .ld file.",
            f" * Family: {self._family.upper()}",
            " */",
            "",
            "MEMORY",
            "{",
            f"    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = {flash}",
            f"    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = {ram}",
            "}",
            "",
            "SECTIONS",
            "{",
            "    /* Model weights in flash (read-only) */",
            "    .timber_model_data : ALIGN(4) {",
            "        _timber_model_data_start = .;",
            "        *(.rodata.timber_*)",
            "        _timber_model_data_end = .;",
            "    } > FLASH",
            "",
            "    /* Inference scratch buffers in RAM */",
            "    .timber_scratch (NOLOAD) : ALIGN(4) {",
            "        *(.timber_scratch)",
            "    } > RAM",
            "}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Override: additional STM32-specific passes
    # ------------------------------------------------------------------

    def emit(self, ir: TimberIR) -> C99Output:
        """Generate STM32 HAL compatible output."""
        output = super().emit(ir)

        # Add CubeMX project-structure hints as comments in header
        model_h = self._add_cubemx_hints(output.model_h)

        return C99Output(
            model_c=output.model_c,
            model_h=model_h,
            model_data_c=output.model_data_c,
            cmakelists=self._emit_stm32_cmake(),
            makefile=output.makefile,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_dma_support(self) -> list[str]:
        """Generate DMA transfer helpers for bulk input data."""
        return [
            "/* --- DMA Support for Bulk Input Transfer --- */",
            "",
            "extern DMA_HandleTypeDef hdma_memtomem;",
            "",
            "/*",
            " * timber_dma_load_input: non-blocking DMA transfer of input",
            " * features from a peripheral buffer to the static input buffer.",
            " */",
            "#define TIMBER_DMA_MAX_FEATURES 4096",
            "#define TIMBER_DMA_TIMEOUT_MS   1000",
            "",
            "static HAL_StatusTypeDef timber_dma_load_input(",
            "    const volatile float* src,",
            "    uint32_t n_features)",
            "{",
            "    if (n_features == 0 || n_features > TIMBER_DMA_MAX_FEATURES) return HAL_ERROR;",
            "    return HAL_DMA_Start(&hdma_memtomem,",
            "        (uint32_t)src,",
            "        (uint32_t)_timber_input_buf,",
            "        n_features * sizeof(float));",
            "}",
            "",
            "/*",
            " * timber_dma_wait: block until DMA transfer completes (finite timeout).",
            " */",
            "static inline HAL_StatusTypeDef timber_dma_wait(void) {",
            "    return HAL_DMA_PollForTransfer(",
            "        &hdma_memtomem, HAL_DMA_FULL_TRANSFER, TIMBER_DMA_TIMEOUT_MS);",
            "}",
            "",
        ]

    def _estimate_stack(self, ir: TimberIR) -> int:
        """Conservative stack-usage estimate in bytes."""
        base = 80  # caller-save + locals + HAL overhead
        max_depth = 32
        for stage in ir.pipeline:
            if hasattr(stage, "max_depth"):
                max_depth = stage.max_depth
        return base + 12 * max_depth

    def _add_cubemx_hints(self, model_h: str) -> str:
        """Add STM32CubeMX integration hints to header."""
        hints = [
            "/*",
            " * STM32CubeMX Integration Guide:",
            " * --------------------------------",
            f" * 1. Target family: {self._family.upper()}",
            f" * 2. Enable {self._timer_instance} as basic timer for inference timing",
            " * 3. Copy model.c, model.h, model_data.c into Core/Src/ and Core/Inc/",
            ' * 4. Add #include "model.h" to your main.c',
            " * 5. Call timber_embedded_init() after HAL_Init() + SystemClock_Config()",
            " * 6. Call timber_embedded_infer(input, output, &ctx) for inference",
            " */",
            "",
        ]
        if self._use_dma:
            hints.insert(-1, f" * 7. Enable DMA (memory-to-memory) for bulk input transfer")
        return "\n".join(hints) + model_h

    def _emit_stm32_cmake(self) -> str:
        """Generate CMakeLists.txt compatible with STM32 CMake toolchain."""
        prefix = self.target.cross_prefix
        flags = self.target.cpu_flags
        lines = [
            "cmake_minimum_required(VERSION 3.16)",
            "project(timber_model C)",
            "",
            f"set(CMAKE_SYSTEM_NAME Generic)",
            f"set(CMAKE_SYSTEM_PROCESSOR {self.target.arch})",
            f"set(CMAKE_C_COMPILER {prefix}gcc)",
            "",
            "# STM32 compile flags",
            f'set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} {flags} -Os '
            f'-ffunction-sections -fdata-sections -Wall -fno-common")',
            'set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} '
            '-Wl,--gc-sections")',
            "",
            "# Sources",
            "add_library(timber_model STATIC",
            "    model.c",
            "    model_data.c",
            ")",
            "",
            "target_include_directories(timber_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})",
            "",
            "target_compile_definitions(timber_model PRIVATE",
            "    TIMBER_NO_HEAP=1",
            "    TIMBER_EMBEDDED=1",
            f"    TIMBER_STM32_FAMILY={self._family.upper()}",
            ")",
        ]
        return "\n".join(lines)
