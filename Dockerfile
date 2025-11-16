# LLaMA-Factory FP8 Test Environment
# Based on NVIDIA PyTorch NGC Container with Transformer Engine support
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    curl \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Clone and install LLaMA-Factory (using fork with Accelerate config support)
RUN git clone --depth 1 -b fix/accelerate-config-support \
    https://github.com/sbhavani/LLaMA-Factory.git && \
    cd LLaMA-Factory && \
    pip install -e ".[deepspeed,metrics]" --no-build-isolation

# Install additional dependencies for FP8 with Transformer Engine
RUN pip install --no-cache-dir \
    transformer-engine[pytorch] \
    deepspeed \
    wandb \
    torchao

# Upgrade torchao to latest version (base image may have outdated version)
RUN pip install --upgrade --no-cache-dir torchao

# Set environment variables for optimal FP8 performance
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV WANDB_DISABLED=true

# Create directories for configs and checkpoints
RUN mkdir -p /workspace/configs /workspace/checkpoints /workspace/scripts

# Copy test configurations
COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/

# Set permissions
RUN chmod +x /workspace/scripts/*.sh

WORKDIR /workspace/LLaMA-Factory

# Default command
CMD ["/bin/bash"]
