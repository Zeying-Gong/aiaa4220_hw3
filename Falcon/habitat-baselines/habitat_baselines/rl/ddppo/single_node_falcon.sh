#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1 
# 设置每个GPU使用16个CPU核心
# CPU_CORES_PER_GPU=1 # 16
TOTAL_GPU=4

# 将总的CPU核心数计算出来
# TOTAL_CPU_CORES=$((CPU_CORES_PER_GPU * TOTAL_GPU))

set -x
# OMP_NUM_THREADS=$CPU_CORES_PER_GPU \
    python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $TOTAL_GPU \
    habitat-baselines/habitat_baselines/run.py \
    --config-name=social_nav_v2/falcon_hm3d_train.yaml \
    > evaluation/falcon/hm3d/train.log 2>&1