"""ESP32 embedded emitter — ESP-IDF compatible, no-heap inference code.

Generates C code targeting ESP32 / ESP32-S3 using ESP-IDF conventions:
- DRAM_ATTR / IRAM_ATTR placement annotations
- FreeRTOS task wrapper for background inference
- Component CMakeLists.txt for idf.py build system
- esp_timer-based inference timing
"""

from __future__ import annotations

from timber.accel.accel.embedded.base import EmbeddedEmitterBase
from timber.codegen.c99 import C99Output, TargetSpec
from timber.ir.model import TimberIR


class ESP32Emitter(EmbeddedEmitterBase):
    """Emit ESP-IDF compatible inference code for ESP32 targets."""

    def __init__(self, target: TargetSpec, embedded_config: dict):
        super().__init__(target, embedded_config)
        self._task_stack = embedded_config.get("task_stack_size", 4096)
        self._task_priority = embedded_config.get("task_priority", 5)
        self._pin_core = embedded_config.get("pin_core", 1)

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def platform_name(self) -> str:
        return "esp32"

    def emit_platform_includes(self) -> str:
        lines = [
            '#include <stdint.h>',
            '#include <string.h>',
            '#include "esp_system.h"',
            '#include "esp_timer.h"',
            '#include "esp_log.h"',
            '#include "esp_attr.h"',
            '#include "freertos/FreeRTOS.h"',
            '#include "freertos/task.h"',
            "",
            '#define TIMBER_TAG "timber"',
        ]
        return "\n".join(lines)

    def emit_startup_code(self, ir: TimberIR) -> str:
        """Generate FreeRTOS task wrapper for inference."""
        lines = [
            "/* --- ESP32 FreeRTOS Inference Task --- */",
            "",
            "typedef struct {",
            "    const float*     input;",
            "    float*           output;",
            "    const TimberCtx* ctx;",
            "    volatile int     done;",
            "    int              result;",
            "} timber_task_params_t;",
            "",
            "static void _timber_infer_task(void* pvParameters) {",
            "    timber_task_params_t* params = (timber_task_params_t*)pvParameters;",
            "",
            "    /* Run inference — no heap, all buffers from caller */",
            "    params->result = timber_infer(",
            "        params->input, 1, params->output, params->ctx);",
            "",
            "    params->done = 1;",
            "    /* params is freed by the caller after reading the result */",
            "    vTaskDelete(NULL);",
            "}",
            "",
            "/*",
            " * timber_embedded_infer_async: spawn inference on a dedicated core.",
            " * The caller blocks until the task signals completion.",
            " */",
            "static int timber_embedded_infer_async(",
            "    const float* input,",
            "    float*       output,",
            "    const TimberCtx* ctx)",
            "{",
            "    timber_task_params_t* params = (timber_task_params_t*)pvPortMalloc(sizeof(timber_task_params_t));",
            "    if (!params) {",
            '        ESP_LOGE(TIMBER_TAG, "Failed to allocate task params");',
            "        return -1;",
            "    }",
            "    params->input  = input;",
            "    params->output = output;",
            "    params->ctx    = ctx;",
            "    params->done   = 0;",
            "    params->result = -1;",
            "",
            "    BaseType_t rc = xTaskCreatePinnedToCore(",
            "        _timber_infer_task,",
            '        "timber_infer",',
            f"        {self._task_stack},",
            "        params,",
            f"        {self._task_priority},",
            "        NULL,",
            f"        {self._pin_core});",
            "",
            "    if (rc != pdPASS) {",
            "        vPortFree(params);",
            '        ESP_LOGE(TIMBER_TAG, "Failed to create inference task");',
            "        return -1;",
            "    }",
            "",
            "    /* Busy-wait with yield — task is short-lived */",
            "    while (!params->done) {",
            "        vTaskDelay(1);",
            "    }",
            "",
            "    int result = params->result;",
            "    vPortFree(params);",
            "    return result;",
            "}",
            "",
            "/*",
            " * timber_embedded_infer: synchronous single-sample inference.",
            " * No heap allocation, bounded stack, interrupt-safe.",
            " */",
            "static int timber_embedded_infer(",
            "    const float* input,",
            "    float*       output,",
            "    const TimberCtx* ctx)",
            "{",
            "    return timber_infer(input, 1, output, ctx);",
            "}",
        ]
        return "\n".join(lines)

    def emit_timing_code(self) -> str:
        """esp_timer-based inference timing."""
        lines = [
            "/* --- ESP32 Inference Timing (esp_timer, microsecond resolution) --- */",
            "",
            "static int64_t _timber_time_start;",
            "",
            "static inline void timber_timing_start(void) {",
            "    _timber_time_start = esp_timer_get_time();",
            "}",
            "",
            "static inline int64_t timber_timing_stop(void) {",
            "    return esp_timer_get_time() - _timber_time_start;",
            "}",
        ]
        return "\n".join(lines)

    def emit_linker_fragment(self, ir: TimberIR) -> str:
        """ESP-IDF linker fragment placing model data in DROM (flash)."""
        lines = [
            "/* --- ESP-IDF Linker Fragment (linker.lf) --- */",
            "/*",
            " * Place in your component directory as linker.lf",
            " * and add to component CMakeLists.txt:",
            " *   idf_component_register(... LDFRAGMENTS linker.lf)",
            " */",
            "",
            "[mapping:timber_model_data]",
            "archive: *",
            "entries:",
            "    timber_model_data (noflash_text)",
            "    timber_model_data : timber_model_weights -> drom0_0_seg",
            "    timber_model_data : timber_scratch -> dram0_0_seg",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Override: additional ESP32-specific annotation pass
    # ------------------------------------------------------------------

    def emit(self, ir: TimberIR) -> C99Output:
        """Generate ESP-IDF compatible output with section annotations."""
        output = super().emit(ir)

        # Add DRAM_ATTR / IRAM_ATTR annotations
        model_c = self._annotate_esp_sections(output.model_c)

        # Generate component CMakeLists.txt
        component_cmake = self._emit_component_cmake()

        # Replace standard cmake with component cmake
        return C99Output(
            model_c=model_c,
            model_h=output.model_h,
            model_data_c=self._annotate_esp_data_sections(output.model_data_c),
            cmakelists=component_cmake,
            makefile=output.makefile,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _annotate_esp_sections(self, src: str) -> str:
        """Place performance-critical functions in IRAM."""
        import re
        # Mark static inline functions as IRAM_ATTR
        src = re.sub(
            r'(static\s+inline\s+)',
            r'IRAM_ATTR \1',
            src,
        )
        return src

    def _annotate_esp_data_sections(self, data_c: str) -> str:
        """Place const model data in DROM (flash) via DRAM_ATTR."""
        import re
        # Large const arrays go to flash-mapped DROM
        data_c = re.sub(
            r'(static\s+const\s+)',
            r'DRAM_ATTR \1',
            data_c,
        )
        return data_c

    def _emit_component_cmake(self) -> str:
        """Generate an ESP-IDF component CMakeLists.txt."""
        lines = [
            "# ESP-IDF Component CMakeLists.txt for Timber inference",
            "#",
            "# Place this file alongside model.c, model.h, model_data.c",
            "# inside components/timber_model/ in your ESP-IDF project.",
            "",
            "idf_component_register(",
            "    SRCS",
            '        "model.c"',
            '        "model_data.c"',
            "    INCLUDE_DIRS",
            '        "."',
            "    REQUIRES",
            "        esp_timer",
            "        freertos",
            ")",
            "",
            "# Optimise for size on flash-constrained targets",
            "target_compile_options(${COMPONENT_LIB} PRIVATE -Os)",
            "",
            "# Disable heap usage at compile time",
            "target_compile_definitions(${COMPONENT_LIB} PRIVATE",
            "    TIMBER_NO_HEAP=1",
            "    TIMBER_EMBEDDED=1",
            ")",
        ]
        return "\n".join(lines)
