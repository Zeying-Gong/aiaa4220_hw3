#!/bin/bash
# Training script for 2x V100 GPUs with DD-PPO
# FIXED VERSION - Uses the patched configuration
# 修复版本 - 使用修复后的配置文件

set -e

echo "=========================================="
echo "Falcon Training on 2x V100 GPUs (FIXED)"
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
export GLOO_SOCKET_IFNAME=lo  # GLOO backend also needs interface specified

# PyTorch distributed settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "Environment variables set:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "  GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo ""

echo "Starting distributed training with 2 GPUs..."
echo "Config: social_nav_v2/falcon_hm3d_train_2v100_optimized.yaml (OPTIMIZED)"
echo ""
echo "Optimizations:"
echo "  ✓ num_environments: 8 -> 12 (better VRAM utilization: 18.6GB -> ~28GB)"
echo "  ✓ num_mini_batch: 4 -> 6"
echo "  ✓ Training speedup: ~30% faster"
echo ""
echo "Fixes applied:"
echo "  ✓ Removed sigmoid from auxiliary losses"
echo "  ✓ Added hesitation penalty (-0.01)"
echo "  ✓ Reduced auxiliary loss scale (0.1 -> 0.05)"
echo "  ✓ Enabled learning rate decay"
echo "  ✓ Increased gradient clipping (0.2 -> 0.5)"
echo ""

# Pre-download ResNet50 ImageNet weights to avoid duplicate downloads
echo "Pre-downloading ResNet50 ImageNet weights..."
python3 -c "
import torch
import torchvision.models as models
print('Downloading ResNet50 pretrained weights...')
_ = models.resnet50(pretrained=True)
print('✓ ResNet50 weights cached successfully')
"
echo ""

# Launch DD-PPO training with 2 processes (one per GPU)
# Using torchrun (recommended) instead of deprecated torch.distributed.launch
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    -m habitat_baselines.run \
    --config-name=social_nav_v2/falcon_hm3d_train_2v100_optimized.yaml

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
