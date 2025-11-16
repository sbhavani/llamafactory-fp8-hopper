#!/bin/bash

# Automated setup script for FP8 training on remote server
set -e

echo "=================================================="
echo "LLaMA-Factory FP8 Setup Script"
echo "=================================================="
echo ""

# Check if running on server with GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✅ NVIDIA drivers detected"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and back in, then run this script again."
    exit 0
fi

echo "✅ Docker detected"

# Check NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Container Toolkit not working. Please install it."
    exit 1
fi

echo "✅ NVIDIA Container Toolkit working"
echo ""

# Build Docker image
echo "Building Docker image (this takes ~10-15 minutes)..."
docker build -t llamafactory-fp8:latest .

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To start training:"
echo ""
echo "1. Start container:"
echo "   docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\"
echo "     --shm-size=16g \\"
echo "     -v \$(pwd)/checkpoints:/workspace/checkpoints \\"
echo "     -v \$(pwd)/configs:/workspace/configs \\"
echo "     -v \$(pwd)/scripts:/workspace/scripts \\"
echo "     -v /tmp:/tmp \\"
echo "     -it llamafactory-fp8:latest bash"
echo ""
echo "2. Inside container, run:"
echo "   bash /workspace/scripts/train_fp8_llamafactory.sh"
echo ""
echo "See LLAMAFACTORY_USER_GUIDE.md for detailed instructions."
echo ""
