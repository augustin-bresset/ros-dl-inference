"""
ROS 2 launch file for the inference node.

Usage:
  ros2 launch ros_rl_inference inference.launch.py config:=/path/to/config.yaml
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("ros_rl_inference")

    default_config = PathJoinSubstitution(
        [pkg_share, "example_pytorch.yaml"]
    )

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Absolute path to the YAML inference config file",
    )

    inference_node = Node(
        package="ros_rl_inference",
        executable="inference_node_ros2",
        name="rl_inference",
        output="screen",
        arguments=["--config", LaunchConfiguration("config")],
        # Highest priority process on the machine
        # Requires CAP_SYS_NICE or running as root
        # additional_env={"SCHED_POLICY": "FIFO", "SCHED_PRIORITY": "80"},
        parameters=[],
        remappings=[],
    )

    return LaunchDescription([
        config_arg,
        inference_node,
    ])
