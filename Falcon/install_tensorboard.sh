#!/bin/bash
# 在Docker容器中永久安装TensorBoard的脚本

echo "======================================"
echo "永久安装TensorBoard到Docker容器"
echo "======================================"

# 方法1: 直接在容器中安装（下次重启容器会丢失）
echo ""
echo "方法1: 临时安装（容器重启后丢失）"
echo "--------------------------------------"
echo "docker exec -it Improve bash"
echo "pip install tensorboard"

# 方法2: 修改Dockerfile并重建镜像（推荐）
echo ""
echo "方法2: 修改Dockerfile永久安装（推荐）"
echo "--------------------------------------"
echo "1. 找到原始Dockerfile或创建新的"
echo "2. 添加: RUN pip install tensorboard"
echo "3. 重建镜像"

# 方法3: 提交容器为新镜像
echo ""
echo "方法3: 提交当前容器为新镜像（最简单）"
echo "--------------------------------------"
echo "# 步骤1: 在容器内安装tensorboard"
echo "docker exec -it Improve bash -c 'pip install tensorboard'"
echo ""
echo "# 步骤2: 提交容器为新镜像"
echo "docker commit Improve quay.io/zeyinggong/robosense_socialnav:v0.7-tensorboard"
echo ""
echo "# 步骤3: 停止并删除旧容器"
echo "docker stop Improve"
echo "docker rm Improve"
echo ""
echo "# 步骤4: 使用新镜像创建容器"
echo "docker run -d --name Improve --gpus all \\"
echo "  -v /home/husrcf/Code/P3/aiaa4220_hw3:/workspace \\"
echo "  quay.io/zeyinggong/robosense_socialnav:v0.7-tensorboard"

echo ""
echo "======================================"
echo "推荐使用方法3，最简单且有效"
echo "======================================"
