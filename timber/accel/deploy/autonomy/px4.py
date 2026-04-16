"""PX4 autopilot module generator for drone ML inference integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_px4_module(
    model_name: str = "timber_model",
    n_features: int = 4,
    n_outputs: int = 1,
    module_name: str = "timber_inference",
    input_topics: list[str] | None = None,
    output_topic: str = "timber_output",
    rate_hz: int = 50,
    output_dir: str | None = None,
) -> dict[str, str]:
    """Generate a PX4 uORB module for ML inference.

    Args:
        model_name: Compiled Timber model name.
        n_features: Number of input features.
        n_outputs: Number of output predictions.
        module_name: PX4 module name.
        input_topics: uORB topics to subscribe (default: vehicle_imu).
        output_topic: uORB topic to publish results.
        rate_hz: Module execution rate in Hz.
        output_dir: If set, write files to this directory.

    Returns:
        Dict mapping relative file paths to content.
    """
    if input_topics is None:
        input_topics = ["vehicle_imu"]

    files: dict[str, str] = {}

    class_name = "".join(w.capitalize() for w in module_name.split("_"))

    # uORB message definition
    files[f"msg/{output_topic}.msg"] = f"""\
uint64 timestamp          # time since system start (microseconds)
float32[{n_outputs}] predictions   # ML model predictions
float32 inference_time_us # inference latency in microseconds
uint32 inference_count    # total inferences since module start
"""

    # Module header
    files[f"{module_name}.h"] = f"""\
#pragma once

#include <px4_platform_common/defines.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/posix.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>

#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/topics/vehicle_imu.h>
#include <uORB/topics/{output_topic}.h>

extern "C" {{
#include "model.h"
}}

class {class_name} : public ModuleBase<{class_name}>, public ModuleParams,
                     public px4::ScheduledWorkItem {{
public:
    {class_name}();
    ~{class_name}() override;

    static int task_spawn(int argc, char *argv[]);
    static int custom_command(int argc, char *argv[]);
    static int print_usage(const char *reason = nullptr);

    bool init();

private:
    void Run() override;

    // Timber model context
    TimberCtx *ctx_{{nullptr}};

    // uORB subscriptions
    uORB::Subscription _vehicle_imu_sub{{ORB_ID(vehicle_imu)}};

    // uORB publication
    uORB::Publication<{output_topic}_s> _output_pub{{ORB_ID({output_topic})}};

    // Buffers
    float _features[{n_features}]{{}};
    float _outputs[{n_outputs}]{{}};

    // Statistics
    uint32_t _inference_count{{0}};

    DEFINE_PARAMETERS(
        (ParamInt<px4::params::TIMBER_RATE>) _param_rate
    )
}};
"""

    # Module source
    files[f"{module_name}.cpp"] = f"""\
#include "{module_name}.h"

#include <px4_platform_common/getopt.h>
#include <px4_platform_common/log.h>
#include <drivers/drv_hrt.h>

{class_name}::{class_name}() :
    ModuleParams(nullptr),
    ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::lp_default)
{{
}}

{class_name}::~{class_name}()
{{
    if (ctx_) {{
        timber_free(ctx_);
        ctx_ = nullptr;
    }}
}}

bool {class_name}::init()
{{
    int rc = timber_init(&ctx_);
    if (rc != TIMBER_OK) {{
        PX4_ERR("timber_init failed: %s", timber_strerror(rc));
        return false;
    }}

    ScheduleOnInterval({int(1_000_000 / rate_hz)}_us);  // {rate_hz} Hz
    PX4_INFO("Timber inference module initialized ({n_features} features, {n_outputs} outputs)");
    return true;
}}

void {class_name}::Run()
{{
    if (should_exit()) {{
        ScheduleClear();
        exit_and_cleanup();
        return;
    }}

    // Read sensor data
    vehicle_imu_s imu;
    if (_vehicle_imu_sub.update(&imu)) {{
        // Extract features from IMU data
        // Map accelerometer + gyroscope to feature vector
        int idx = 0;
        if (idx < {n_features}) _features[idx++] = imu.delta_velocity[0];
        if (idx < {n_features}) _features[idx++] = imu.delta_velocity[1];
        if (idx < {n_features}) _features[idx++] = imu.delta_velocity[2];
        if (idx < {n_features}) _features[idx++] = imu.delta_angle[0];
        if (idx < {n_features}) _features[idx++] = imu.delta_angle[1];
        if (idx < {n_features}) _features[idx++] = imu.delta_angle[2];
        // Zero remaining features
        for (; idx < {n_features}; idx++) {{
            _features[idx] = 0.0f;
        }}

        // Run inference
        hrt_abstime start = hrt_absolute_time();
        int rc = timber_infer_single(_features, _outputs, ctx_);
        float elapsed_us = (float)(hrt_absolute_time() - start);

        if (rc == TIMBER_OK) {{
            _inference_count++;

            // Publish result
            {output_topic}_s result{{}};
            result.timestamp = hrt_absolute_time();
            for (int i = 0; i < {n_outputs}; i++) {{
                result.predictions[i] = _outputs[i];
            }}
            result.inference_time_us = elapsed_us;
            result.inference_count = _inference_count;
            _output_pub.publish(result);
        }} else {{
            PX4_WARN("Inference failed: %s", timber_strerror(rc));
        }}
    }}
}}

int {class_name}::task_spawn(int argc, char *argv[])
{{
    auto *instance = new {class_name}();
    if (!instance) {{
        PX4_ERR("alloc failed");
        return PX4_ERROR;
    }}

    _object.store(instance);
    _task_id = task_id_is_work_queue;

    if (!instance->init()) {{
        delete instance;
        _object.store(nullptr);
        _task_id = -1;
        return PX4_ERROR;
    }}

    return PX4_OK;
}}

int {class_name}::custom_command(int argc, char *argv[])
{{
    return print_usage("unknown command");
}}

int {class_name}::print_usage(const char *reason)
{{
    if (reason) {{
        PX4_WARN("%s", reason);
    }}

    PRINT_MODULE_DESCRIPTION(
        R"DESCR_STR(
### Description
Timber ML inference module. Runs compiled tree-ensemble model
on sensor data and publishes predictions via uORB.
)DESCR_STR");

    PRINT_MODULE_USAGE_NAME("{module_name}", "estimator");
    PRINT_MODULE_USAGE_COMMAND("start");
    PRINT_MODULE_USAGE_DEFAULT_COMMANDS();

    return 0;
}}

extern "C" __EXPORT int {module_name}_main(int argc, char *argv[])
{{
    return {class_name}::main(argc, argv);
}}
"""

    # CMakeLists.txt for PX4 module
    files["CMakeLists.txt"] = f"""\
px4_add_module(
    MODULE modules__{module_name}
    MAIN {module_name}
    SRCS
        {module_name}.cpp
        model/model.c
        model/model_data.c
    INCLUDES
        model/
    DEPENDS
    )
"""

    # Kconfig
    files["Kconfig"] = f"""\
menuconfig MODULES_{module_name.upper()}
    bool "{module_name}"
    default n
    ---help---
        Timber ML inference module for PX4.
"""

    if output_dir:
        out = Path(output_dir) / module_name
        for rel_path, content in files.items():
            fp = out / rel_path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)

    return files
