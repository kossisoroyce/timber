"""ROS 2 node generator for ML inference integration."""

from __future__ import annotations

from pathlib import Path


def generate_ros2_package(
    model_name: str = "timber_model",
    n_features: int = 4,
    n_outputs: int = 1,
    input_topic: str = "/sensor_data",
    output_topic: str = "/inference_result",
    node_name: str = "timber_inference_node",
    package_name: str = "timber_inference",
    output_dir: str | None = None,
) -> dict[str, str]:
    """Generate a complete ROS 2 package for ML inference.

    Args:
        model_name: Name of the compiled Timber model.
        n_features: Number of input features.
        n_outputs: Number of output predictions.
        input_topic: ROS topic to subscribe for sensor data.
        output_topic: ROS topic to publish inference results.
        node_name: ROS node name.
        package_name: ROS package name.
        output_dir: If set, write files to this directory.

    Returns:
        Dict mapping relative file paths to their content.
    """
    files: dict[str, str] = {}

    # package.xml
    files["package.xml"] = f"""\
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>0.1.0</version>
  <description>Timber Accelerate ML inference node</description>
  <maintainer email="dev@timber.dev">Timber Team</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
"""

    # CMakeLists.txt
    files["CMakeLists.txt"] = f"""\
cmake_minimum_required(VERSION 3.8)
project({package_name})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Timber compiled model
add_library(timber_model STATIC
  model/model.c
  model/model_data.c
)
target_include_directories(timber_model PUBLIC model/)

# Inference node
add_executable({node_name}
  src/{node_name}.cpp
)
target_link_libraries({node_name} timber_model m)
ament_target_dependencies({node_name} rclcpp std_msgs sensor_msgs)

install(TARGETS {node_name}
  DESTINATION lib/${{PROJECT_NAME}}
)

install(DIRECTORY launch/
  DESTINATION share/${{PROJECT_NAME}}/launch
)

ament_package()
"""

    # Node source
    files[f"src/{node_name}.cpp"] = f"""\
#include <chrono>
#include <climits>
#include <functional>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

extern "C" {{
#include "model.h"
}}

using namespace std::chrono_literals;

class TimberInferenceNode : public rclcpp::Node {{
public:
    TimberInferenceNode() : Node("{node_name}") {{
        // Initialize Timber model context
        int rc = timber_init(&ctx_);
        if (rc != TIMBER_OK) {{
            RCLCPP_FATAL(this->get_logger(), "Failed to init Timber model: %s",
                         timber_strerror(rc));
            throw std::runtime_error("timber_init failed");
        }}

        // Subscriber for sensor data
        sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "{input_topic}", 10,
            std::bind(&TimberInferenceNode::on_sensor_data, this, std::placeholders::_1));

        // Publisher for inference results
        pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "{output_topic}", 10);

        RCLCPP_INFO(this->get_logger(),
                     "Timber inference node ready: %d features -> %d outputs",
                     TIMBER_N_FEATURES, TIMBER_N_OUTPUTS);
    }}

    ~TimberInferenceNode() {{
        if (ctx_) timber_free(ctx_);
    }}

private:
    void on_sensor_data(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {{
        size_t total = msg->data.size();
        if (total < (size_t)TIMBER_N_FEATURES) {{
            RCLCPP_WARN(this->get_logger(),
                        "Input size %zu < %d features, skipping",
                        total, TIMBER_N_FEATURES);
            return;
        }}

        if (total > (size_t)INT_MAX || total % TIMBER_N_FEATURES != 0) return;
        int n_samples = (int)(total / TIMBER_N_FEATURES);

        std::vector<float> outputs(n_samples * TIMBER_N_OUTPUTS);

        auto start = this->now();
        int rc = timber_infer(msg->data.data(), n_samples, outputs.data(), ctx_);
        auto elapsed = (this->now() - start).nanoseconds() / 1000.0;

        if (rc != TIMBER_OK) {{
            RCLCPP_ERROR(this->get_logger(), "Inference failed: %s",
                         timber_strerror(rc));
            return;
        }}

        // Publish results
        auto result = std_msgs::msg::Float32MultiArray();
        result.data = std::move(outputs);
        pub_->publish(result);

        RCLCPP_DEBUG(this->get_logger(),
                     "Inference: %d samples in %.1f us", n_samples, elapsed);
        total_inferences_ += n_samples;
    }}

    TimberCtx* ctx_ = nullptr;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
    int64_t total_inferences_ = 0;
}};

int main(int argc, char* argv[]) {{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TimberInferenceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}}
"""

    # Launch file
    files[f"launch/{node_name}_launch.py"] = f"""\
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{package_name}',
            executable='{node_name}',
            name='{node_name}',
            output='screen',
            parameters=[],
        ),
    ])
"""

    if output_dir:
        out = Path(output_dir) / package_name
        for rel_path, content in files.items():
            fp = out / rel_path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)

    return files
