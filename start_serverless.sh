#!/bin/bash
# Serverless startup script
# This script initializes the environment and starts the handler

set -e

echo "================================================"
echo "Diffusion-Pipe Serverless Worker Starting..."
echo "================================================"

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
if [ -n "$TCMALLOC" ]; then
    export LD_PRELOAD="${TCMALLOC}"
    echo "Using tcmalloc: $TCMALLOC"
fi

# Determine the network volume path
# In serverless, it's /runpod-volume; in pods, it's /workspace
# The handler will add /diffusion_pipe_working_folder to this base path
if [ -d "/runpod-volume" ]; then
    export NETWORK_VOLUME="/runpod-volume"
    echo "Using RunPod serverless volume: /runpod-volume"
elif [ -d "/workspace" ]; then
    export NETWORK_VOLUME="/workspace"
    echo "Using RunPod pod volume: /workspace"
else
    export NETWORK_VOLUME="/tmp"
    echo "No persistent volume found, using /tmp (data will not persist)"
fi

# The WORKING_DIR is where all training files will be stored
export WORKING_DIR="$NETWORK_VOLUME/diffusion_pipe_working_folder"
mkdir -p "$WORKING_DIR"
echo "NETWORK_VOLUME: $NETWORK_VOLUME"
echo "WORKING_DIR: $WORKING_DIR"

# Create required directory structure
mkdir -p "$WORKING_DIR/image_dataset_here"
mkdir -p "$WORKING_DIR/video_dataset_here"
mkdir -p "$WORKING_DIR/training_outputs"
mkdir -p "$WORKING_DIR/models"
mkdir -p "$WORKING_DIR/logs"

# Copy diffusion_pipe to working directory if not already there
if [ ! -d "$WORKING_DIR/diffusion_pipe" ]; then
    echo "Setting up diffusion_pipe in working directory..."
    cp -r /diffusion_pipe "$WORKING_DIR/"
fi

# Copy Captioning scripts if not already there
if [ ! -d "$WORKING_DIR/Captioning" ]; then
    echo "Setting up Captioning scripts..."
    cp -r /serverless/Captioning "$WORKING_DIR/"

    # Make scripts executable
    chmod +x "$WORKING_DIR/Captioning/JoyCaption/JoyCaptionRunner.sh" 2>/dev/null || true
    chmod +x "$WORKING_DIR/Captioning/video_captioner.sh" 2>/dev/null || true
fi

# Ensure logs directory exists for JoyCaption
mkdir -p "$WORKING_DIR/logs"

# Ensure examples directory exists
mkdir -p "$WORKING_DIR/diffusion_pipe/examples"

# GPU detection and info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Check if flash-attn is installed, if not try the prebuilt wheel
if ! python3 -c "import flash_attn" 2>/dev/null; then
    echo "Installing flash-attn from prebuilt wheel..."

    FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"

    cd /tmp
    WHEEL_NAME=$(basename "$FLASH_ATTN_WHEEL_URL")

    if wget -q -O "$WHEEL_NAME" "$FLASH_ATTN_WHEEL_URL" 2>&1; then
        pip install "$WHEEL_NAME" 2>&1 || echo "Warning: flash-attn installation failed"
        rm -f "$WHEEL_NAME"
    fi
    cd /
fi

# Verify key packages
echo ""
echo "Verifying installed packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import runpod; print(f'RunPod SDK: {runpod.__version__}')"
python3 -c "import flash_attn; print('flash-attn: OK')" 2>/dev/null || echo "flash-attn: Not available (training may still work)"

echo ""
echo "================================================"
echo "Starting serverless handler..."
echo "================================================"

# Start the handler
exec python3 -u /serverless/handler.py
