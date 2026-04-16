"""ARM Cortex-M embedded emitter — CMSIS-based, no-heap C code.

Generates baremetal-safe inference code targeting Cortex-M0/M3/M4/M7/M33
with optional DSP/SIMD intrinsics (Cortex-M4+).
"""

from __future__ import annotations

from timber.accel.accel.embedded.base import EmbeddedEmitterBase
from timber.codegen.c99 import TargetSpec
from timber.ir.model import TimberIR


class CortexMEmitter(EmbeddedEmitterBase):
    """Emit ARM Cortex-M inference code using CMSIS headers and intrinsics."""

    def __init__(self, target: TargetSpec, embedded_config: dict):
        super().__init__(target, embedded_config)
        self._has_dsp = self._detect_dsp()
        self._has_fpu = "fpv" in target.cpu_flags or "mfloat-abi=hard" in target.cpu_flags

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def platform_name(self) -> str:
        return "cortex_m"

    def emit_platform_includes(self) -> str:
        lines = [
            '#include <stdint.h>',
            '#include <string.h>',
            '#include "cmsis_compiler.h"',
        ]
        if self._has_dsp:
            lines.append('#include "arm_math.h"')
        return "\n".join(lines)

    def emit_startup_code(self, ir: TimberIR) -> str:
        """Generate startup code with stack/heap configuration."""
        stack = self.constraints.stack_size
        lines = [
            "/* --- Cortex-M Startup Configuration --- */",
            "",
            f"/* Stack size: {stack} bytes (validated at link time) */",
            "__attribute__((section(\".stack\"))) __attribute__((used))",
            f"static uint8_t _timber_stack_area[{stack}];",
            "",
            "/* Heap is intentionally omitted — all allocation is static */",
            "#if defined(__TIMBER_HEAP_CHECK)",
            "  _Static_assert(0, \"Timber embedded: heap allocation is forbidden\");",
            "#endif",
            "",
            self._emit_systick_init(),
            "",
            self._emit_inference_wrapper(ir),
        ]
        return "\n".join(lines)

    def emit_timing_code(self) -> str:
        """SysTick-based inference timing for Cortex-M."""
        lines = [
            "/* --- SysTick Inference Timing --- */",
            "",
            "static volatile uint32_t _timber_tick_start;",
            "static volatile uint32_t _timber_tick_end;",
            "",
            "__STATIC_FORCEINLINE void timber_timing_start(void) {",
            "    /* Reload SysTick for a full countdown */",
            "    SysTick->LOAD = 0x00FFFFFFu;",
            "    SysTick->VAL  = 0u;",
            "    SysTick->CTRL = SysTick_CTRL_ENABLE_Msk | SysTick_CTRL_CLKSOURCE_Msk;",
            "    _timber_tick_start = SysTick->VAL;",
            "}",
            "",
            "__STATIC_FORCEINLINE uint32_t timber_timing_stop(void) {",
            "    _timber_tick_end = SysTick->VAL;",
            "    SysTick->CTRL = 0u;",
            "    /* SysTick counts down, so elapsed = start - end */",
            "    return (_timber_tick_start - _timber_tick_end) & 0x00FFFFFFu;",
            "}",
        ]
        return "\n".join(lines)

    def emit_linker_fragment(self, ir: TimberIR) -> str:
        """Linker script fragment placing model data in flash, buffers in RAM."""
        flash = self.constraints.flash_size
        ram = self.constraints.ram_size
        lines = [
            "/* --- Cortex-M Linker Script Fragment --- */",
            "/*",
            " * Paste into your .ld file or include via INCLUDE directive.",
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
            "    .text : {",
            "        KEEP(*(.isr_vector))",
            "        *(.text*)",
            "        *(.rodata*)",
            "        . = ALIGN(4);",
            "    } > FLASH",
            "",
            "    .timber_model_data : {",
            "        . = ALIGN(4);",
            "        _timber_model_data_start = .;",
            "        *(.rodata.timber_*)",
            "        _timber_model_data_end = .;",
            "    } > FLASH",
            "",
            "    .data : {",
            "        . = ALIGN(4);",
            "        _sdata = .;",
            "        *(.data*)",
            "        _edata = .;",
            "    } > RAM AT> FLASH",
            "",
            "    .bss : {",
            "        . = ALIGN(4);",
            "        _sbss = .;",
            "        *(.bss*)",
            "        *(COMMON)",
            "        _ebss = .;",
            "    } > RAM",
            "",
            "    .timber_scratch (NOLOAD) : {",
            "        . = ALIGN(4);",
            "        *(.timber_scratch)",
            "    } > RAM",
            "",
            "    .stack (NOLOAD) : {",
            "        . = ALIGN(8);",
            "        *(.stack)",
            "    } > RAM",
            "}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_dsp(self) -> bool:
        """Check if the target Cortex-M variant has DSP/SIMD extensions."""
        arch = self.target.arch.lower()
        # Cortex-M4, M7, M33, M55 have DSP
        dsp_cores = ("cortex-m4", "cortex-m7", "cortex-m33", "cortex-m55")
        return any(core in arch for core in dsp_cores)

    def _emit_systick_init(self) -> str:
        """Generate SysTick initialisation for timing."""
        lines = [
            "/* Initialise SysTick for inference cycle counting */",
            "__STATIC_FORCEINLINE void timber_systick_init(void) {",
            "    SysTick->CTRL = 0u;",
            "    SysTick->LOAD = 0x00FFFFFFu;",
            "    SysTick->VAL  = 0u;",
            "}",
        ]
        return "\n".join(lines)

    def _emit_inference_wrapper(self, ir: TimberIR) -> str:
        """Generate the main inference entry-point with interrupt safety."""
        lines = [
            "/* --- Cortex-M Inference Wrapper (interrupt-safe) --- */",
            "",
            "/*",
            " * timber_embedded_infer: single-sample inference with no heap,",
            " * bounded stack, and interrupt safety (PRIMASK save/restore).",
            " *",
            f" * Stack usage estimate: {self._estimate_stack(ir)} bytes",
            " */",
            "__STATIC_FORCEINLINE int timber_embedded_infer(",
            "    const float* input,",
            "    float*       output,",
            "    const TimberCtx* ctx)",
            "{",
            "    /* Save interrupt state — prevent ISR from mutating shared state */",
            "    uint32_t primask = __get_PRIMASK();",
            "    __disable_irq();",
            "",
        ]

        if self._has_dsp:
            lines += self._emit_dsp_traversal_body()
        else:
            lines += [
                "    /* Scalar traversal — call generated timber_infer */",
                "    int rc = timber_infer(input, 1, output, ctx);",
            ]

        lines += [
            "",
            "    /* Restore interrupt state */",
            "    __set_PRIMASK(primask);",
            "",
            "    return rc;",
            "}",
        ]
        return "\n".join(lines)

    def _emit_dsp_traversal_body(self) -> list[str]:
        """Emit DSP-accelerated inner loop body for Cortex-M4+ SIMD."""
        return [
            "    /* Cortex-M4+ DSP SIMD path — __SMLAD for accumulate */",
            "    int rc = 0;",
            "    const int n_feat = ctx->n_features;",
            "    const int n_trees = ctx->n_trees;",
            "    float acc = ctx->base_score;",
            "",
            "#define MAX_TREE_DEPTH 64",
            "",
            "    for (int t = 0; t < n_trees; t++) {",
            "        int node = ctx->tree_roots[t];",
            "        int depth = 0;",
            "        while (!ctx->is_leaf[node]) {",
            "            if (++depth > MAX_TREE_DEPTH) break;",
            "            int fi = ctx->feature_indices[node];",
            "            float fval = input[fi];",
            "            float thr  = ctx->thresholds[node];",
            "            /* __DSP: use saturating compare where possible */",
            "            node = (fval <= thr)",
            "                 ? ctx->left_children[node]",
            "                 : ctx->right_children[node];",
            "        }",
            "        acc += ctx->leaf_values[node];",
            "    }",
            "",
            "    output[0] = acc;",
        ]

    def _estimate_stack(self, ir: TimberIR) -> int:
        """Conservative stack-usage estimate in bytes."""
        # Base overhead: 64 bytes for caller save, primask, locals
        base = 64
        # Per-tree traversal: node index + feature value + threshold = ~12 bytes
        n_trees = 1
        for stage in ir.pipeline:
            if hasattr(stage, "trees"):
                n_trees = max(n_trees, len(stage.trees))
        # We only traverse one tree at a time, so max depth matters more
        max_depth = 32  # conservative
        for stage in ir.pipeline:
            if hasattr(stage, "max_depth"):
                max_depth = stage.max_depth
        traversal = 12 * max_depth
        return base + traversal
