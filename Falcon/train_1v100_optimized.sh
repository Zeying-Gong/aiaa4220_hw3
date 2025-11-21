#!/bin/bash
# Training script for single V100 GPU (optimized)
# Use this if distributed training has issues

set -e

echo "=========================================="
echo "Falcon Training on 1x V100 GPU (Optimized)"
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

echo "Environment variables set:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

echo "Starting single-GPU training..."
echo "Config: social_nav_v2/falcon_hm3d_train_1v100_optimized.yaml"
echo ""

# Launch training on single GPU
python -u -m habitat_baselines.run \
    --config-name=social_nav_v2/falcon_hm3d_train_1v100_optimized.yaml

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="