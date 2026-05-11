FROM blopausore/torchsparse:1.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app/off-road-dg-lss:/app/ros-rl-inference/src:${PYTHONPATH}

# Recompile torchsparse CUDA extensions (le container de base a les sources
# mais les .so ne sont pas installés dans site-packages)
RUN cd /workspace/torchsparse && \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9" \
    python3 setup.py build_ext --inplace

# Dépendances supplémentaires
RUN pip install --no-cache-dir \
    rosbags \
    pyyaml \
    pandas \
    open3d \
    scikit-learn

WORKDIR /app

CMD ["bash"]
