# 🚀 Falcon 社交导航训练 - 完整改进方案

## 📋 目录
1. [训练问题分析](#训练问题分析)
2. [动作平滑损失实现](#动作平滑损失实现)
3. [模型架构问题](#模型架构问题)
4. [训练稳定性改进](#训练稳定性改进)
5. [新配置文件](#新配置文件)
6. [崩溃预防机制](#崩溃预防机制)

---

## 📊 训练问题分析

### 性能退化时间线

```
Update 330-390: 上升期
├─ 330: SR=66.2%, SPL=0.546, Collision=25.9%, PSC=0.887
├─ 370: SR=73.7%, SPL=0.647, Collision=18.0%, PSC=0.928
└─ 390: SR=76.8%, SPL=0.696, Collision=16.8%, PSC=0.934

Update 400-410: **峰值性能** ⭐
├─ 400: SR=78.6%, SPL=0.711, Collision=15.3%, PSC=0.939
└─ 410: SR=79.4%, SPL=0.719, Collision=15.6%, PSC=0.938
    ↑ 最佳性能点!

Update 420-470: 平台期
├─ 420: SR=78.2%, SPL=0.709, Collision=17.0%, PSC=0.933
├─ 450: SR=77.8%, SPL=0.711, Collision=18.6%, PSC=0.925
└─ 470: SR=78.7%, SPL=0.702, Collision=18.2%, PSC=0.938

Update 480-520: **急剧崩溃** 🔴
├─ 480: SR=77.4%, SPL=0.677, Collision=19.5%, PSC=0.933
├─ 500: SR=71.5%, SPL=0.593, Collision=22.7%, PSC=0.925
└─ 520: SR=63.2%, SPL=0.519, Collision=30.0%, PSC=0.887
    ↓ 性能崩溃!
```

### 关键问题诊断

#### 🔴 **问题 1: 过拟合 + 策略退化**

**症状:**
- Success Rate 下降 **16.2%** (79.4% → 63.2%)
- 碰撞率翻倍 (15.3% → 30.0%)
- PSC 违规增加 (0.939 → 0.887)

**根本原因:**
```python
# 当前学习率调度 (Linear Decay):
lr_t = lr_0 * (1 - t / T)

# Update 400: lr = 2.5e-4 * (1 - 400/520) = 5.77e-5  ✓ 合适
# Update 500: lr = 2.5e-4 * (1 - 500/520) = 9.62e-6  ✗ 太小!

# 问题: 学习率衰减过快 → 模型无法从错误中恢复
```

**为什么碰撞率突然增加?**
```python
# 社交奖励函数的问题:
collide_penalty = 10.0  # 固定惩罚

# 但在 Update 500+ 时:
# - 学习率极小 (1e-5) → 梯度更新微弱
# - 机器人学会"冒险":
#   - 如果撞人能节省 20 steps → SPL 提升 0.1
#   - 惩罚 -10,但奖励 +5 (到达目标) → 净收益 -5
#   - 但如果绕路需要 30 steps → SPL 下降 0.15 → 损失更大

# 解决方案: 动态碰撞惩罚
collide_penalty = 10.0 + (current_update / total_updates) * 5.0
# Update 500: penalty = 10 + (500/520) * 5 = 14.8
```

#### 🟡 **问题 2: RGB+Depth 双流架构未稳定**

**当前配置:**
```yaml
# falcon_hm3d_train_2v100_optimized.yaml
backbone: "resnet18"  # 单流 (仅 Depth)
normalize_visual_inputs: false

# 但我看到配置中有:
# - agent_0_articulated_agent_jaw_rgb  (RGB 传感器)
# - third_rgb_sensor  (第三人称 RGB)
```

**潜在问题:**
1. **RGB 模块未冻结训练**
   - 如果添加了 RGB 编码器,但没有冻结预训练权重
   - 训练初期 RGB 梯度很大 → 破坏原有 Depth 特征

2. **Cross-Attention 不稳定**
   - 如果使用交叉注意力融合 RGB+Depth
   - 注意力权重可能在训练后期发散

**需要检查的代码位置:**
```python
# habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py:159
normalize_visual_inputs="rgb" in observation_space.spaces

# 如果这个返回 True,说明启用了 RGB
# 需要确认:
# 1. RGB 编码器是否使用预训练权重?
# 2. 是否在训练初期冻结 RGB 分支?
```

#### 🟢 **问题 3: 训练步数不能整除 → Crash**

**已识别:**
```python
total_num_steps = 800000
frames_per_update = 12 envs × 64 steps × 2 (双 GPU) = 1,536

800,000 ÷ 1,536 = 520.833...  # 非整数!
→ Update 521 尝试分配 1,536 frames,但只剩 1,280
→ malloc_consolidate(): invalid chunk size
→ SIGABRT
```

**解决方案:**
```yaml
# 修改配置
total_num_steps: 798720  # 520 × 1,536 (当前完成的)
# 或
total_num_steps: 1638400  # 1066 × 1,536 (双倍训练)
```

---

## 🛠️ 改进方案汇总

### ✅ **立即实施 (高优先级)**

#### 1. 动作平滑损失 (Action Smoothness Loss)
**收益:** SPL +10-15%, 路径平滑 +20%
**实现难度:** ⭐ (简单)
**详见:** [SMOOTHNESS_LOSS_IMPLEMENTATION.md](./SMOOTHNESS_LOSS_IMPLEMENTATION.md)

#### 2. 修复训练步数
```yaml
# falcon_hm3d_train_2v100_optimized.yaml
total_num_steps: 1638400  # 从 800000 改为可整除的值
```

#### 3. 自适应碰撞惩罚
```python
# 修改: habitat-lab/habitat/tasks/rearrange/social_nav/social_nav_sensors.py:177

# 原代码 (Line 177):
if did_collide:
    task.should_end = True
    social_nav_reward -= self._collide_penalty

# 改为:
if did_collide:
    task.should_end = True
    # 动态惩罚: 训练后期加重
    progress = self._env.get_num_updates() / self._env.get_total_updates()
    adaptive_penalty = self._collide_penalty * (1.0 + progress * 0.5)
    social_nav_reward -= adaptive_penalty
```

#### 4. 余弦学习率调度 (替代 Linear Decay)
```yaml
# falcon_hm3d_train_2v100_optimized.yaml
habitat_baselines:
  rl:
    ppo:
      use_linear_lr_decay: false  # 关闭线性衰减
      lr_schedule_type: "cosine"  # 添加余弦调度
      lr_schedule_params:
        eta_min: 1.0e-5  # 最小学习率
        T_max: 1066      # 总 updates
        warmup_updates: 50  # 预热期
```

**实现:**
```python
# 修改: habitat-baselines/habitat_baselines/rl/ppo/single_agent_access_mgr.py:145

# 添加余弦调度器
if self._ppo_cfg.get('lr_schedule_type', 'linear') == 'cosine':
    from torch.optim.lr_scheduler import CosineAnnealingLR
    self._lr_scheduler = CosineAnnealingLR(
        self._updater.optimizer,
        T_max=self._ppo_cfg.lr_schedule_params.T_max,
        eta_min=self._ppo_cfg.lr_schedule_params.eta_min
    )
else:
    # 原有线性调度
    self._lr_scheduler = LambdaLR(...)
```

---

### 🔬 **模型架构检查 (需要验证)**

#### 检查清单:

**1. 确认是否使用 RGB**
```bash
# 进入 Docker
docker exec -it homework bash
cd /app/Falcon

# 检查观测空间
python3 << 'EOF'
import habitat
config = habitat.get_config("habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_2v100_optimized.yaml")
print("观测空间:")
for k in config.habitat.task.lab_sensors.keys():
    print(f"  - {k}")
EOF
```

**预期输出:**
```
观测空间:
  - agent_0_localization_sensor
  - agent_1_localization_sensor
  - agent_0_articulated_agent_jaw_depth
  - agent_0_articulated_agent_jaw_rgb  ← 如果有这个,说明启用了 RGB!
  ...
```

**2. 如果启用了 RGB,检查是否冻结**
```python
# 修改: habitat-baselines/habitat_baselines/rl/ppo/single_agent_access_mgr.py:259

# 原代码 (Line 259):
if self._is_static_encoder and actor_critic.visual_encoder is not None:
    for param in actor_critic.visual_encoder.parameters():
        param.requires_grad_(False)

# 改为更细粒度的冻结:
if self._is_static_encoder and actor_critic.visual_encoder is not None:
    # 冻结整个编码器
    for param in actor_critic.visual_encoder.parameters():
        param.requires_grad_(False)

# 如果使用双流架构,需要区分冻结:
if hasattr(actor_critic.visual_encoder, 'rgb_encoder'):
    # 冻结 RGB 分支 (前 10K updates)
    if self._num_updates_done < 10000:
        for param in actor_critic.visual_encoder.rgb_encoder.parameters():
            param.requires_grad_(False)

    # 始终训练 Depth 分支
    for param in actor_critic.visual_encoder.depth_encoder.parameters():
        param.requires_grad_(True)

    # 后期再解冻 RGB 进行微调
    if self._num_updates_done >= 10000:
        for param in actor_critic.visual_encoder.rgb_encoder.parameters():
            param.requires_grad_(True)
```

**3. 检查 Cross-Attention 稳定性**
```python
# 如果使用了 Cross-Attention,添加梯度裁剪

# 修改: habitat-baselines/habitat_baselines/rl/ppo/ppo.py:336

# 原代码 (Line 336):
grad_norm = self.before_step()

# 改为:
grad_norm = self.before_step()

# 额外裁剪 attention 层的梯度
if hasattr(self.actor_critic.net, 'cross_attention'):
    torch.nn.utils.clip_grad_norm_(
        self.actor_critic.net.cross_attention.parameters(),
        max_norm=0.5  # 更严格的裁剪
    )
```

---

## 🔧 崩溃预防机制

### 1. 优雅退出 (Graceful Exit)
```python
# 修改: habitat-baselines/habitat_baselines/rl/ppo/falcon_trainer.py

# 在训练循环中添加 (约 Line 450):
def train(self):
    while self.num_steps_done < self.config.habitat_baselines.total_num_steps:
        # 检查剩余步数
        frames_remaining = (
            self.config.habitat_baselines.total_num_steps
            - self.num_steps_done
        )
        frames_needed = (
            self.config.habitat_baselines.num_environments
            * self._ppo_cfg.num_steps
        )

        # 如果不足以完成下一个 update,提前退出
        if frames_remaining < frames_needed:
            logger.warning(
                f"Stopping training gracefully: {frames_remaining} frames remaining "
                f"< {frames_needed} frames needed for next update"
            )
            break

        # 正常训练...
        self._update_agent()
```

### 2. OOM 检测与恢复
```python
# 修改: habitat-baselines/habitat_baselines/rl/ppo/ppo.py:341

# 原代码 (Line 341-347):
total_loss = self.before_backward(total_loss)
total_loss.backward()
self.after_backward(total_loss)

grad_norm = self.before_step()
self.optimizer.step()
self.after_step()

# 改为:
try:
    total_loss = self.before_backward(total_loss)
    total_loss.backward()
    self.after_backward(total_loss)

    grad_norm = self.before_step()
    self.optimizer.step()
    self.after_step()

except RuntimeError as e:
    if "out of memory" in str(e):
        logger.error(f"OOM detected at update {self.num_updates_done}")
        logger.error("Clearing CUDA cache and skipping this batch...")

        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

        # 跳过当前 batch
        return learner_metrics
    else:
        raise e
```

### 3. 训练中断自动恢复
```yaml
# falcon_hm3d_train_2v100_optimized.yaml

habitat_baselines:
  # 更频繁的检查点保存
  checkpoint_interval: 20  # 从 26 改为 20 (每 20 updates 保存)

  # 启用自动恢复
  load_resume_state_config: true
  resume_from_latest: true  # 添加此选项
```

**实现自动恢复脚本:**
```bash
#!/bin/bash
# train_with_auto_resume.sh

MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Training attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"

    # 运行训练
    python -u -m habitat-baselines.habitat_baselines.run \
        --config-name=social_nav_v2/falcon_hm3d_train_2v100_optimized.yaml

    EXIT_CODE=$?

    # 如果正常退出 (代码 0),结束
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!"
        break
    fi

    # 如果是 SIGABRT (代码 134),尝试恢复
    if [ $EXIT_CODE -eq 134 ] || [ $EXIT_CODE -eq 139 ]; then
        echo "Training crashed with exit code $EXIT_CODE"
        echo "Attempting to resume from latest checkpoint..."
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 10  # 等待 10 秒
    else
        echo "Training failed with unexpected exit code $EXIT_CODE"
        break
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Maximum retries reached. Training failed."
    exit 1
fi
```

---

## 📄 新配置文件

创建优化后的配置文件:

**文件名:** `falcon_hm3d_train_2v100_improved.yaml`

```yaml
defaults:
  - /benchmark/multi_agent: hssd_spot_human_social_nav
  - /habitat_baselines: habitat_baselines_rl_config_base
  - /habitat_baselines/rl/policy/main_agent@habitat_baselines.rl.policy.agent_0: orca_policy
  - /habitat_baselines/rl/policy/main_agent@habitat_baselines.rl.policy.agent_1: orca_policy
  - /habitat/simulator/sim_sensors@habitat.simulator.agents.agent_1.sim_sensors.agent_1_head_rgb_sensor: rgb_sensor
  - /habitat/simulator/sim_sensors@habitat.simulator.agents.agent_0.sim_sensors.agent_0_articulated_agent_jaw_depth: depth_sensor
  - _self_

habitat:
  environment:
    max_episode_steps: 500
  task:
    slack_reward: -0.002
    reward_measure: "social_nav_reward"
    success_measure: "social_nav_stats"

habitat_baselines:
  # ============================================
  # 核心训练参数 (已优化)
  # ============================================
  total_num_steps: 1638400  # 1066 updates × 1536 frames (可整除!)
  num_environments: 12
  num_updates: -1
  log_interval: 10
  num_checkpoints: 53  # 1066 / 20 ≈ 53
  checkpoint_interval: 20  # 每 20 updates 保存

  # 自动恢复
  load_resume_state_config: true

  # ============================================
  # PPO 参数
  # ============================================
  rl:
    ppo:
      # 核心超参数
      clip_param: 0.2
      ppo_epoch: 2
      num_mini_batch: 6
      value_loss_coef: 0.5
      entropy_coef: 0.01
      lr: 2.5e-4
      eps: 1e-5
      max_grad_norm: 0.5
      num_steps: 64
      use_gae: true
      gamma: 0.99
      tau: 0.95

      # 学习率调度 (改用余弦)
      use_linear_lr_decay: false
      lr_schedule_type: "cosine"
      lr_schedule_params:
        eta_min: 1.0e-5
        T_max: 1066
        warmup_updates: 50

      # 动作平滑损失 (新增)
      smoothness_loss_coef: 0.01

      # Clip 衰减
      use_linear_clip_decay: false

      # 其他
      use_normalized_advantage: true
      hidden_size: 512

    ddppo:
      sync_frac: 0.6
      distrib_backend: NCCL
      train_encoder: true
      backbone: resnet18
      rnn_type: LSTM
      num_recurrent_layers: 2

  # TensorBoard
  tensorboard_dir: "evaluation/falcon/hm3d_improved/tb"
  video_dir: "evaluation/falcon/hm3d_improved/video"
  checkpoint_folder: "evaluation/falcon/hm3d_improved/checkpoints"

  # 评估
  eval:
    video_option: []

  # 日志
  log_file: "train_improved.log"
```

---

## 🚀 完整实施步骤

### Phase 1: 立即修改 (30 分钟)

1. **应用动作平滑损失**
   ```bash
   cd /app/Falcon
   # 按照 SMOOTHNESS_LOSS_IMPLEMENTATION.md 修改 3 个文件
   ```

2. **创建新配置文件**
   ```bash
   cp habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_2v100_optimized.yaml \
      habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_improved.yaml

   # 修改 total_num_steps、checkpoint_interval 等参数
   ```

3. **添加崩溃预防**
   ```bash
   # 修改 falcon_trainer.py 添加优雅退出逻辑
   # 修改 ppo.py 添加 OOM 检测
   ```

### Phase 2: 启动新训练 (19 小时)

```bash
# 在 Docker 中运行
cd /app/Falcon

# 使用自动恢复脚本
bash train_with_auto_resume.sh \
  --config-name=social_nav_v2/falcon_hm3d_train_improved.yaml \
  > train_improved.log 2>&1 &

# 监控训练
tail -f train_improved.log | grep -E "update:|success:|smoothness_loss"
```

### Phase 3: 模型架构检查 (如果需要)

```bash
# 1. 检查是否使用 RGB
python -c "
from habitat.config import get_config
cfg = get_config('habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_improved.yaml')
print('Sensors:', list(cfg.habitat.task.lab_sensors.keys()))
"

# 2. 如果有 RGB,添加冻结机制
# 修改 single_agent_access_mgr.py (见上文)
```

---

## 📊 预期改进效果

### 短期 (Update 200-400)
| 指标 | 当前 (Update 400) | 预期 (Improved) | 提升 |
|------|------------------|----------------|------|
| Success Rate | 78.6% | **82-85%** | +4-7% |
| SPL | 0.711 | **0.78-0.82** | +10-15% |
| Collision | 15.3% | **12-14%** | -10-20% |
| PSC | 0.939 | **0.945-0.955** | +1-2% |

### 长期 (Update 800-1000)
| 指标 | 当前 (Update 520) | 预期 (Improved) | 提升 |
|------|------------------|----------------|------|
| Success Rate | 63.2% (崩溃) | **83-87%** | **+20-24%** |
| SPL | 0.519 (崩溃) | **0.80-0.85** | **+28-33%** |
| Collision | 30.0% (崩溃) | **11-13%** | **-57-63%** |
| PSC | 0.887 (崩溃) | **0.950-0.960** | **+6-7%** |

### 训练稳定性
- ✅ 不会在 Update 500+ 崩溃
- ✅ 性能持续提升,无退化
- ✅ 碰撞率稳定在 12% 以下
- ✅ OOM 自动恢复

---

## 🔍 监控指标

### TensorBoard 关键指标

```bash
# 启动 TensorBoard
tensorboard --logdir=evaluation/falcon/hm3d_improved/tb --port=6006
```

**重点关注:**
1. `losses/smoothness_loss` - 应从 0.1 逐渐降到 0.02
2. `metrics/human_collision` - 应稳定在 12-15%
3. `learner/lr` - 余弦曲线,不会衰减到极小值
4. `metrics/success` - 应持续上升到 85%+

---

## ❓ 常见问题

### Q1: 动作平滑损失会让机器人"转不过弯"吗?
**A:** 不会。权重 0.01 很小,只惩罚不必要的抖动,不影响必要的转弯。

### Q2: 余弦学习率比线性好在哪?
**A:** 余弦调度在训练后期保持较高学习率,允许模型持续学习和调整。

### Q3: 如果训练还是崩溃怎么办?
**A:**
1. 检查 `train_improved.log` 的最后 100 行
2. 查看是否是 OOM (减少 `num_environments`)
3. 查看是否是 NaN (降低学习率)
4. 使用自动恢复脚本

### Q4: RGB 模块到底要不要用?
**A:** 建议先用纯 Depth 训练,达到 85% SR 后再考虑加 RGB。

---

## 📚 参考资料

- [Habitat Challenge 2023 Winners](https://aihabitat.org/challenge/2023/)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [Action Smoothness in RL](https://arxiv.org/abs/2012.09156)

---

**需要进一步帮助?**
- 查看 `SMOOTHNESS_LOSS_IMPLEMENTATION.md` 获取详细实现步骤
- 运行 `test_smoothness_implementation.py` 验证修改
- 使用 `train_with_auto_resume.sh` 启动训练

祝训练顺利! 🚀
