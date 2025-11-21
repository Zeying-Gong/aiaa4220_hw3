#!/bin/bash
# 在宿主机上启动TensorBoard监控训练
# 这个脚本在宿主机上运行，不需要进入Docker

echo "=========================================="
echo "TensorBoard 监控脚本 (宿主机版本)"
echo "=========================================="
echo ""

# TensorBoard日志路径（宿主机上的路径）
LOGDIR="/home/husrcf/Code/P3/aiaa4220_hw3/Falcon/evaluation/falcon"

echo "可用的训练日志目录："
echo "  1. hm3d_2v100_dual (当前正在崩溃的训练)"
echo "  2. hm3d_2v100_optimized (新的优化训练，如果已启动)"
echo "  3. hm3d_2v100 (旧训练)"
echo "  4. hm3d_1v100_opt (单GPU训练)"
echo "  5. hm3d (最早的训练)"
echo ""

# 检查Python和TensorBoard
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 找不到python3"
    echo "请先安装: sudo apt install python3 python3-pip"
    exit 1
fi

if ! python3 -c "import tensorboard" 2>/dev/null; then
    echo "⚠️  TensorBoard未安装在宿主机上"
    echo "正在安装 TensorBoard..."
    pip3 install tensorboard
    echo ""
fi

echo "✅ TensorBoard已就绪"
echo ""
echo "启动TensorBoard服务器..."
echo "日志目录: $LOGDIR"
echo "访问地址: http://localhost:6006"
echo ""
echo "提示："
echo "  - 可以选择监控特定目录，例如:"
echo "    tensorboard --logdir=$LOGDIR/hm3d_2v100_optimized/tb"
echo "  - 或监控所有训练:"
echo "    tensorboard --logdir=$LOGDIR"
echo ""
echo "按 Ctrl+C 停止TensorBoard"
echo "=========================================="
echo ""

# 默认监控当前正在运行的训练（hm3d_2v100_dual）
# 你可以修改这里来选择不同的目录
CURRENT_LOG="$LOGDIR/hm3d_2v100_dual/tb"

if [ -d "$LOGDIR/hm3d_2v100_optimized/tb" ]; then
    echo "检测到新的优化训练日志，切换到: hm3d_2v100_optimized"
    CURRENT_LOG="$LOGDIR/hm3d_2v100_optimized/tb"
fi

tensorboard --logdir="$CURRENT_LOG" --host=0.0.0.0 --port=6006
