#!/bin/bash
# Training script for 2x V100 GPUs with DD-PPO
# This script launches distributed training across both GPUs

set -e

echo "=========================================="
echo "Falcon Training on 2x V100 GPUs"
echo "=========================================="
echo ""

# Check GPU availability
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Set environment variables for optimal performance
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Enable NCCL optimizations for V100
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export NCCL_SOCKET_IFNAME=lo  # Use loopback for single-node

# PyTorch distributed settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "Environment variables set:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  NCCL_DEBUG=$NCCL_DEBUG"
echo ""

# Change to Falcon directory
cd Falcon

echo "Starting distributed training with 2 GPUs..."
echo "Config: social_nav_v2/falcon_hm3d_train_2v100.yaml"
echo ""

# Launch DD-PPO training with 2 processes (one per GPU)
python -u -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    -m habitat-baselines.habitat_baselines.run \
    --config-name=social_nav_v2/falcon_hm3d_train_2v100.yaml

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="