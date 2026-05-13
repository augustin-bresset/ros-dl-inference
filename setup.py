"""Standalone pip-installable setup for the core Python library (no ROS required)."""

from setuptools import find_packages, setup

setup(
    name="ros_dl_inference",
    version="0.1.0",
    description="Optimized model inference for ROS 1 and ROS 2",
    author="Augustin Bresset",
    author_email="augustin.bresset@gmail.com",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pyyaml>=6.0",
    ],
    extras_require={
        "pytorch": ["torch>=2.0"],
        "onnx": ["onnxruntime-gpu>=1.16"],
        "jax": ["jax[cuda]>=0.4", "flax", "orbax-checkpoint"],
        "tensorrt": ["tensorrt>=8.6", "pycuda"],
        "dev": [
            "pytest>=7.0",
            "pytest-benchmark",
        ],
    },
    entry_points={
        "console_scripts": [
            "rl_inference_ros1=ros_dl_inference.nodes.inference_node_ros1:main",
            "rl_inference_ros2=ros_dl_inference.nodes.inference_node_ros2:main",
        ]
    },
)
