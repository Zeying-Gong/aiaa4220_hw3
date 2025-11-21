# 训练崩溃问题诊断与修复报告

## 📊 问题诊断

### 症状
- **更新300**突然崩溃：success从59% → 2%，num_steps从97 → 13
- 机器人学会了"立即停止"策略，几乎不移动

### 根本原因分析

#### 🔴 原因1: 辅助任务损失使用sigmoid导致梯度消失
```python
# 问题代码 (falcon/auxiliary_tasks.py)
ori_loss = self.loss_fn(logits, target)
sigmoid_loss = torch.sigmoid(ori_loss)  # ❌ 导致梯度饱和
loss = self.loss_scale * sigmoid_loss
```

**影响**：
- 当loss很大时，sigmoid(loss) → 1.0，梯度消失
- 辅助任务无法正常学习，反而干扰主任务

#### 🔴 原因2: 缺少"犹豫"惩罚
机器人发现"不动"可以避免所有风险：
- 不会撞人（-0.5惩罚）
- 不会撞墙（-0.25惩罚）
- 不会靠近人类（-0.025惩罚）

但系统没有惩罚"原地不动"的行为。

#### 🔴 原因3: 梯度检查点与辅助任务冲突
DualStreamFpnAttentionEncoder使用了gradient checkpointing，可能与辅助任务的梯度计算冲突。

---

## ✅ 修复方案

### 修复1: 移除sigmoid，使用clamping
**文件**: `falcon/auxiliary_tasks.py`

```python
# ✅ 修复后
ori_loss = self.loss_fn(logits, target)
loss = self.loss_scale * torch.clamp(ori_loss, max=5.0)  # 直接裁剪
```

**应用到**：
- PeopleCounting (第93-97行)
- GuessHumanPosition (第176-181行)
- FutureTrajectoryPrediction (第249-254行)

### 修复2: 添加"犹豫"惩罚
**文件**: `falcon/additional_metric.py`

```python
# ✅ 新增Component 6 (第323-331行)
if self._prev_robot_pos is not None and distance_to_target > self._allow_distance:
    movement = np.linalg.norm(robot_pos - self._prev_robot_pos)
    if movement < 0.05:  # 移动小于5cm
        social_nav_reward += self._hesitation_penalty  # -0.01惩罚
```

**配置参数** (第524行):
```python
hesitation_penalty: float = -0.01
```

### 修复3: 降低辅助任务权重
**文件**: `habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_2v100_fixed.yaml`

```yaml
auxiliary_losses:
  people_counting:
    loss_scale: 0.05  # 从0.1降低
  guess_human_position:
    loss_scale: 0.05
  future_trajectory_prediction:
    loss_scale: 0.05
```

### 修复4: 启用学习率衰减 + 增加梯度裁剪
```yaml
ppo:
  lr: 1.5e-4  # 降低初始学习率
  max_grad_norm: 0.5  # 从0.2增加到0.5
  use_linear_lr_decay: True  # ✅ 启用衰减
```

### 修复5: 禁用梯度检查点（可选）
如果训练仍不稳定，可以修改 `resnet_policy.py`:

```python
def forward(self, rgb, depth):
    # 禁用checkpoint，增加稳定性
    return self._forward_impl(rgb, depth)
```

---

## 🚀 重新开始训练步骤

### 步骤1: 停止当前训练
```bash
# 进入Docker容器
docker exec -it Improve bash

# 找到训练进程
ps aux | grep habitat_baselines.run

# 杀死进程
kill <PID>
```

### 步骤2: 备份当前checkpoint（可选）
```bash
cd /workspace/Falcon
cp -r evaluation/falcon/hm3d_2v100_dual evaluation/falcon/hm3d_2v100_dual_backup
```

### 步骤3: 使用修复后的配置重新训练
```bash
cd /workspace/Falcon

# 使用修复后的配置
torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=29500 \
  -m habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_train_2v100_fixed.yaml
```

### 步骤4: 监控训练
```bash
# 实时查看日志
tail -f train.log

# 查看tensorboard
tensorboard --logdir evaluation/falcon/hm3d_2v100_fixed/tb
```

---

## 📈 预期结果

修复后的训练应该：

1. **前100更新**：success率稳定在40-60%
2. **100-300更新**：逐渐提升，不会崩溃
3. **300+更新**：继续稳定提升或保持

**关键指标监控**：
- `num_steps` 应该保持在 100-200 之间（不应该降到10-20）
- `success` 应该持续提升或稳定（不应该突然跌到0.01）
- `reward` 应该保持正值（不应该变成负数）

---

## 📝 修改文件清单

1. ✅ `falcon/auxiliary_tasks.py` - 移除sigmoid，3处修改
2. ✅ `falcon/additional_metric.py` - 添加犹豫惩罚，3处修改
3. ✅ `habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_2v100_fixed.yaml` - 新配置文件

**可选修改**：
4. `habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py` - 禁用梯度检查点（如果仍不稳定）

---

## 🎯 关键改进

| 问题 | 修复 | 效果 |
|------|------|------|
| 辅助任务梯度消失 | 移除sigmoid，使用clamping | 辅助任务正常学习 |
| 机器人"冻结"策略 | 添加犹豫惩罚-0.01 | 鼓励机器人移动 |
| 辅助任务主导主任务 | loss_scale: 0.1→0.05 | 主任务权重提升 |
| 训练后期不稳定 | 启用LR衰减 | 后期更稳定 |
| 梯度爆炸 | max_grad_norm: 0.2→0.5 | 梯度更稳定 |

---

生成时间: 2025-11-20
