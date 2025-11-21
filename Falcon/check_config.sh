#!/bin/bash
# 在Docker容器内运行此脚本来诊断配置

echo "======================================"
echo "配置诊断脚本"
echo "======================================"

CONFIG_FILE="habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_mini_junwei.yaml"

echo ""
echo "1. 检查RGB相机是否启用:"
echo "--------------------------------------"
grep -A 2 "obs_keys:" $CONFIG_FILE | head -5

echo ""
echo "2. 检查Backbone配置:"
echo "--------------------------------------"
grep "backbone:" $CONFIG_FILE

echo ""
echo "3. 检查预训练权重配置:"
echo "--------------------------------------"
grep -A 2 "pretrained" $CONFIG_FILE | grep -E "pretrained:|pretrained_weights:"

echo ""
echo "4. 检查辅助任务损失权重:"
echo "--------------------------------------"
grep -A 1 "loss_scale:" $CONFIG_FILE

echo ""
echo "5. 检查学习率和优化器:"
echo "--------------------------------------"
grep -E "lr:|clip_param:|entropy_coef:" $CONFIG_FILE

echo ""
echo "6. 检查checkpoint文件:"
echo "--------------------------------------"
ls -lh evaluation/falcon/hm3d/checkpoints/

echo ""
echo "7. 检查训练日志最后20行:"
echo "--------------------------------------"
tail -20 train.log

echo ""
echo "======================================"
echo "诊断完成"
echo "======================================"
