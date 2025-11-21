#!/usr/bin/env python3
import torch
import sys

print("=" * 60)
print("模型权重诊断脚本")
print("=" * 60)

checkpoint_path = "evaluation/falcon/hm3d/checkpoints/latest.pth"
pretrained_path = "pretrained_model/pretrained_mini.pth"

try:
    print(f"\n加载checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(ckpt, list):
        state_dict = ckpt[0]['state_dict']
    else:
        state_dict = ckpt['state_dict']

    print(f"✅ Checkpoint加载成功")
    print(f"   总参数数量: {len(state_dict)}")

    print("\n🔍 检查模型架构关键层:")

    visual_encoder_keys = [k for k in state_dict.keys() if 'visual_encoder' in k]
    print(f"\n1. Visual Encoder层数: {len(visual_encoder_keys)}")

    rgb_keys = [k for k in visual_encoder_keys if 'rgb' in k]
    depth_keys = [k for k in visual_encoder_keys if 'depth' in k]
    fpn_keys = [k for k in visual_encoder_keys if 'fpn' in k or 'lateral' in k or 'smooth' in k]
    se_keys = [k for k in visual_encoder_keys if 'se_mlp' in k or 'fc_compress' in k]

    print(f"   - RGB Stream: {len(rgb_keys)} 层")
    if len(rgb_keys) > 0:
        print(f"     示例: {rgb_keys[0]}")

    print(f"   - Depth Stream: {len(depth_keys)} 层")
    if len(depth_keys) > 0:
        print(f"     示例: {depth_keys[0]}")

    print(f"   - FPN层: {len(fpn_keys)} 层")
    if len(fpn_keys) > 0:
        print(f"     示例: {fpn_keys[0]}")

    print(f"   - SE Module: {len(se_keys)} 层")
    if len(se_keys) > 0:
        print(f"     示例: {se_keys[0]}")

    print("\n2. 检查是否使用dual_stream_fpn:")
    if len(rgb_keys) > 0 and len(fpn_keys) > 0:
        print("   ✅ 确认使用DualStreamFpnAttentionEncoder")
    else:
        print("   ❌ 未检测到DualStream架构，可能使用标准ResNet")

    print("\n3. 辅助任务模块:")
    aux_keys = [k for k in state_dict.keys() if 'aux_loss_modules' in k]
    print(f"   辅助任务参数: {len(aux_keys)} 个")

    aux_modules = set([k.split('.')[1] for k in aux_keys if len(k.split('.')) > 1])
    for module in aux_modules:
        module_keys = [k for k in aux_keys if f'.{module}.' in k]
        print(f"   - {module}: {len(module_keys)} 参数")

    print("\n4. 参数统计:")
    total_params = sum(p.numel() for p in state_dict.values())
    visual_params = sum(state_dict[k].numel() for k in visual_encoder_keys)
    aux_params = sum(state_dict[k].numel() for k in aux_keys)

    print(f"   总参数量: {total_params:,}")
    print(f"   Visual Encoder: {visual_params:,} ({visual_params/total_params*100:.1f}%)")
    print(f"   辅助任务: {aux_params:,} ({aux_params/total_params*100:.1f}%)")
    print(f"   其他: {total_params-visual_params-aux_params:,}")

except FileNotFoundError:
    print(f"❌ 未找到checkpoint: {checkpoint_path}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 加载失败: {e}")
    sys.exit(1)

try:
    print(f"\n" + "=" * 60)
    print(f"对比预训练模型: {pretrained_path}")
    print("=" * 60)

    pretrained = torch.load(pretrained_path, map_location='cpu')
    if isinstance(pretrained, list):
        pre_state = pretrained[0]['state_dict']
    else:
        pre_state = pretrained['state_dict']

    print(f"✅ 预训练模型加载成功")
    print(f"   总参数数量: {len(pre_state)}")

    pre_visual_keys = [k for k in pre_state.keys() if 'visual_encoder' in k]
    print(f"   Visual Encoder层数: {len(pre_visual_keys)}")

    print("\n5. 权重匹配度检查:")
    matched = 0
    mismatched = 0
    new_keys = 0

    for key in visual_encoder_keys:
        if key in pre_state:
            if state_dict[key].shape == pre_state[key].shape:
                matched += 1
            else:
                mismatched += 1
                print(f"   ⚠️  维度不匹配: {key}")
                print(f"      当前: {state_dict[key].shape}, 预训练: {pre_state[key].shape}")
        else:
            new_keys += 1

    print(f"\n   匹配: {matched} 层")
    print(f"   不匹配: {mismatched} 层")
    print(f"   新增: {new_keys} 层")

    if new_keys > len(visual_encoder_keys) * 0.5:
        print("\n   ⚠️  警告: 超过50%的层是新增的（未预训练）")
        print("   这可能导致训练不稳定！")

except FileNotFoundError:
    print(f"❌ 未找到预训练模型: {pretrained_path}")
except Exception as e:
    print(f"❌ 加载预训练模型失败: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
