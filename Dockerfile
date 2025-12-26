# LLaMA-Factory FP8 Test Environment
# Based on NVIDIA PyTorch NGC Container with Transformer Engine support
FROM nvcr.io/nvidia/pytorch:25.12-py3

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

# Clone and install LLaMA-Factory (using fork with FP8/Accelerate fixes)
RUN git clone --depth 1 -b fix/accelerate-config-support \
    https://github.com/sbhavani/LLaMA-Factory.git && \
    cd LLaMA-Factory && \
    pip install -e ".[deepspeed,metrics]" --no-build-isolation

# Install additional dependencies
RUN pip install --no-cache-dir \
    deepspeed \
    wandb

# Set environment variables for optimal FP8 performance
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV WANDB_DISABLED=true
ENV TOKENIZERS_PARALLELISM=false

# Create directories for configs and checkpoints
RUN mkdir -p /workspace/configs /workspace/checkpoints /workspace/scripts

# Copy test configurations and scripts
COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/

# Set permissions
RUN chmod +x /workspace/scripts/*.sh

# Verify installation
RUN python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')" && \
    python -c "import llamafactory; print('LLaMA-Factory OK')" && \
    python -c "from accelerate import __version__; print(f'Accelerate: {__version__}')"

WORKDIR /workspace/LLaMA-Factory

# Default command
CMD ["/bin/bash"]
