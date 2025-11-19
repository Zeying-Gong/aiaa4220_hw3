# GPU Optimization Guide for Falcon Training

## Problem Analysis

### Current Configuration Issues
Your default mini config (`falcon_hm3d_train_mini_junwei.yaml`) only uses **~8GB VRAM** and **<15% GPU utilization** because:

1. **Small batch size**: Only 4 parallel environments with 2 mini-batches
2. **Small model**: ResNet18 + LSTM128 is computationally lightweight
3. **CPU bottleneck**: Habitat simulation runs on CPU, creating imbalance
4. **Single GPU**: Not leveraging your second V100
5. **Single-threaded PyTorch**: `force_torch_single_threaded: True` limits parallelism

### Hardware Available
- **2x NVIDIA V100** (32GB VRAM each)
- **Total capacity**: 64GB VRAM
- **Current usage**: ~8GB on 1 GPU (12.5% utilization)

---

## Optimization Strategies

### Configuration Comparison

| Parameter | Mini Config | Optimized 2-GPU | Optimized 1-GPU | Full Model |
|-----------|-------------|-----------------|-----------------|------------|
| **num_environments** | 4 | 16 | 12 | 8 |
| **backbone** | resnet18 | resnet50 | resnet50 | resnet50 |
| **hidden_size** | 128 | 512 | 512 | 512 |
| **num_mini_batch** | 2 | 4 | 4 | 2 |
| **ppo_epoch** | 2 | 4 | 4 | 2 |
| **lr** | 1.5e-4 | 2.0e-4 | 2.0e-4 | 2.5e-4 |
| **GPUs** | 1 | 2 (DD-PPO) | 1 | 1-8 |
| **Expected VRAM** | 8GB | 28GB/GPU | 28GB | 24GB |
| **Expected Speedup** | 1x | 4-6x | 2-3x | 2x |

### Key Optimizations

#### 1. Increase Parallel Environments
```yaml
num_environments: 16  # Up from 4
```
- More environments = more samples per update
- Better GPU utilization through larger batch processing
- Reduces CPU-GPU idle time

#### 2. Upgrade Model Architecture
```yaml
backbone: resnet50      # Up from resnet18 (4x more compute)
hidden_size: 512        # Up from 128 (4x larger LSTM)
```
- ResNet50 has 4x more parameters and compute than ResNet18
- Larger LSTM increases memory bandwidth usage
- More compute = better GPU saturation

#### 3. Increase Batch Size
```yaml
num_mini_batch: 4       # Up from 2
ppo_epoch: 4            # Up from 2
```
- Effective batch size: 16 envs × 128 steps ÷ 4 batches = 512 samples/batch
- More gradient updates per rollout
- Better GPU compute utilization

#### 4. Enable Multi-GPU Training (DD-PPO)
```yaml
force_distributed: True
distrib_backend: NCCL
```
- Distributes environments across 2 GPUs
- Each GPU processes 8 environments independently
- Gradients synchronized via NCCL

#### 5. Enable Multi-Threading
```yaml
force_torch_single_threaded: False
```
- Allows PyTorch to use multiple CPU threads
- Better CPU utilization for data loading and preprocessing

---

## Usage Instructions

### Option 1: Distributed Training (2x V100) - RECOMMENDED

**Expected Performance:**
- VRAM usage: ~28GB per GPU
- GPU utilization: 70-90%
- Training speed: 4-6x faster than mini config
- Time to 800K steps: ~4-5 hours (vs 20 hours)

**Launch Command:**
```bash
cd /Users/zhaoj/Project/aiaa4220_hw3
./train_2v100.sh
```

Or manually:
```bash
cd Falcon
python -u -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    -m habitat-baselines.habitat_baselines.run \
    --config-name=social_nav_v2/falcon_hm3d_train_2v100.yaml
```

**Monitor Training:**
```bash
# Terminal 1: Watch GPU usage
watch -n 1 nvidia-smi

# Terminal 2: TensorBoard
tensorboard --logdir=Falcon/evaluation/falcon/hm3d_2v100/tb
```

---

### Option 2: Single GPU Optimized (1x V100)

**Expected Performance:**
- VRAM usage: ~28GB
- GPU utilization: 60-80%
- Training speed: 2-3x faster than mini config
- Time to 800K steps: ~7-8 hours

**Launch Command:**
```bash
cd Falcon
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=social_nav_v2/falcon_hm3d_train_1v100_optimized.yaml
```

---

### Option 3: Original Mini Config (Baseline)

**For comparison or debugging:**
```bash
cd Falcon
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=social_nav_v2/falcon_hm3d_train_mini_junwei.yaml
```

---

## Monitoring and Debugging

### Check GPU Utilization
```bash
# Real-time monitoring
nvidia-smi dmon -s u

# Detailed stats every 2 seconds
watch -n 2 nvidia-smi
```

### Expected Metrics

**Good GPU Utilization:**
- GPU Util: 70-95%
- Memory Usage: 25-30GB / 32GB
- Temperature: 60-80°C
- Power: 200-300W (V100 max is 300W)

**Poor GPU Utilization (indicates issues):**
- GPU Util: <30%
- Memory Usage: <10GB
- Indicates CPU bottleneck or config issues

### TensorBoard Monitoring
```bash
tensorboard --logdir=Falcon/evaluation/falcon/hm3d_2v100/tb --port=6006
```

**Key Metrics to Watch:**
- `perf/fps`: Frames per second (higher is better)
  - Mini config: ~500-800 FPS
  - Optimized 2-GPU: ~2000-4000 FPS
- `reward`: Episode reward (should increase over time)
- `metrics/success`: Success rate
- `learner/value_loss`: Should decrease and stabilize

---

## Troubleshooting

### Issue: OOM (Out of Memory) Error

**Solution 1: Reduce environments**
```yaml
num_environments: 12  # Down from 16
```

**Solution 2: Reduce batch size**
```yaml
num_mini_batch: 2     # Down from 4
ppo_epoch: 2          # Down from 4
```

**Solution 3: Use gradient accumulation**
```yaml
num_mini_batch: 8     # More batches = smaller per-batch size
```

### Issue: Low GPU Utilization Still

**Check CPU bottleneck:**
```bash
htop  # Check if CPUs are maxed out
```

**Increase CPU workers:**
```yaml
num_environments: 20  # Even more environments
```

**Enable async execution:**
```yaml
use_double_buffered_sampler: True
```

### Issue: Distributed Training Fails

**Check NCCL:**
```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # Disable peer-to-peer if issues
```

**Fallback to single GPU:**
Use `falcon_hm3d_train_1v100_optimized.yaml` instead

### Issue: Training Unstable (Loss Spikes)

**Reduce learning rate:**
```yaml
lr: 1.5e-4  # Down from 2.0e-4
```

**Enable gradient clipping:**
```yaml
max_grad_norm: 0.2  # Already enabled
```

**Reduce batch size:**
```yaml
num_mini_batch: 2
ppo_epoch: 2
```

---

## Performance Benchmarks

### Expected Training Times (to 800K steps)

| Configuration | Time | GPU Util | VRAM/GPU | FPS |
|---------------|------|----------|----------|-----|
| Mini (baseline) | ~20h | 15% | 8GB | 600 |
| 1-GPU Optimized | ~7h | 70% | 28GB | 1800 |
| 2-GPU DD-PPO | ~4h | 85% | 28GB | 3500 |
| Full Model (8 envs) | ~10h | 60% | 24GB | 1200 |

*Times are approximate and depend on CPU performance*

---

## Advanced Tuning

### For Maximum Throughput
```yaml
num_environments: 20
num_mini_batch: 8
ppo_epoch: 2
use_double_buffered_sampler: True
force_torch_single_threaded: False
```

### For Maximum Model Quality
```yaml
num_environments: 8
num_mini_batch: 2
ppo_epoch: 4
lr: 1.5e-4
hidden_size: 512
backbone: resnet50
```

### For Memory-Constrained Systems
```yaml
num_environments: 8
num_mini_batch: 4
hidden_size: 256
backbone: resnet34  # Middle ground
```

---

## Files Created

1. **`Falcon/habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_2v100.yaml`**
   - Optimized config for 2x V100 GPUs with DD-PPO
   - 16 environments, ResNet50, LSTM512

2. **`Falcon/habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train_1v100_optimized.yaml`**
   - Optimized config for single V100
   - 12 environments, ResNet50, LSTM512

3. **`train_2v100.sh`**
   - Convenient launch script for distributed training
   - Sets optimal environment variables

4. **`GPU_OPTIMIZATION_GUIDE.md`** (this file)
   - Comprehensive documentation

---

## Next Steps

1. **Start with 2-GPU config** (recommended):
   ```bash
   ./train_2v100.sh
   ```

2. **Monitor GPU usage** in another terminal:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check TensorBoard** after ~10 minutes:
   ```bash
   tensorboard --logdir=Falcon/evaluation/falcon/hm3d_2v100/tb
   ```

4. **Adjust if needed**:
   - If OOM: Reduce `num_environments` to 12
   - If low GPU util: Increase `num_environments` to 20
   - If unstable: Reduce `lr` to 1.5e-4

5. **Process checkpoint for evaluation**:
   ```bash
   cd Falcon
   python process_ckp_for_eval.py \
       evaluation/falcon/hm3d_2v100/checkpoints/ckpt.15.pth \
       evaluation/falcon/hm3d_2v100/checkpoints/eval.pth
   ```

---

## Questions?

- Check GPU memory: `nvidia-smi`
- Check training logs: `tail -f Falcon/evaluation/falcon/hm3d_2v100/train.log`
- Monitor FPS in TensorBoard: Should be 2000-4000 FPS for 2-GPU config
- Expected convergence: ~400K-600K steps for good performance