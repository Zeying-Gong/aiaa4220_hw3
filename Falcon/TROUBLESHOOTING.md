# Troubleshooting Guide

## Issue 1: `--local-rank` argument error (FIXED)

**Error:**
```
run.py: error: unrecognized arguments: --local-rank=0
```

**Cause:**
The old `torch.distributed.launch` passes `--local-rank` argument, but Habitat's run.py doesn't accept it.

**Solution:**
Updated `train_2v100.sh` to use `torchrun` instead:
```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m habitat_baselines.run \
    --config-name=social_nav_v2/falcon_hm3d_train_2v100.yaml
```

Also enabled `force_distributed: True` in the config.

---

## Testing Strategy

### Step 1: Test Single GPU First
```bash
cd /app/Falcon
bash train_1v100_optimized.sh
```

This will verify:
- Config is correct
- Model loads properly
- Training loop works
- No memory issues

**Expected output:**
- GPU utilization: 60-80%
- VRAM usage: ~28GB
- FPS: ~1800

### Step 2: If Single GPU Works, Try Distributed
```bash
cd /app/Falcon
bash train_2v100.sh
```

**Expected output:**
- Both GPUs at 70-90% utilization
- ~28GB VRAM per GPU
- FPS: ~3500

---

## Common Issues

### Issue: Out of Memory

**Solution 1:** Reduce environments
Edit `falcon_hm3d_train_2v100.yaml`:
```yaml
num_environments: 12  # Down from 16
```

**Solution 2:** Reduce batch size
```yaml
num_mini_batch: 2  # Down from 4
ppo_epoch: 2       # Down from 4
```

### Issue: Low GPU Utilization

**Solution:** Increase environments
```yaml
num_environments: 20  # Up from 16
```

### Issue: Training Unstable

**Solution:** Reduce learning rate
```yaml
lr: 1.5e-4  # Down from 2.0e-4
```

---

## Quick Fixes

### If distributed training keeps failing:

**Option 1: Use single GPU optimized config**
```bash
bash train_1v100_optimized.sh
```
- Still 2-3x faster than mini config
- More stable

**Option 2: Reduce complexity**
Edit `falcon_hm3d_train_2v100.yaml`:
```yaml
num_environments: 8   # Reduce from 16
num_mini_batch: 2     # Reduce from 4
ppo_epoch: 2          # Reduce from 4
```

---

## Monitoring Commands

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor TensorBoard
tensorboard --logdir=evaluation/falcon/hm3d_2v100/tb --port=6006
```
