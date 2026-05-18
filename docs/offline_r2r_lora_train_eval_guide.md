# 离线训练/测试改动说明（R2R + LoRA）

本文档整理当前离线训练/测试改动与使用方式，覆盖数据位置、代码功能、启动脚本、参数配置、输出指标与注意事项。

更新时间：2026-04-16

## 1. 改动清单

本次新增文件：

- `scripts/train/qwenvl_train/offline_r2r_lora_sft_eval.py`
  - 离线数据 SFT 训练 + Trainer eval + 生成式动作评测（generation eval）统一入口。
- `scripts/train/qwenvl_train/run_offline_r2r_quicktest.sh`
  - 快速验证脚本（少量样本 + 少量 step）。
- `scripts/train/qwenvl_train/run_offline_r2r_train.sh`
  - 常规离线训练脚本（参数可通过环境变量覆盖）。

未覆盖旧代码路径；原有训练/评测入口保持不变。

## 2. 数据集位置与格式

当前使用的离线数据路径：

- 训练集：`/home/ubuntu/dataset/VLN-Trajectory-Data/R2R`
- 验证集：`/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen`

脚本期望的数据结构：

```text
<root>/
  annotations.json
  images/
    <episode_id>/
      rgb/
        000.jpg
        001.jpg
        ...
```

`annotations.json` 中每条 episode 主要字段：

- `video`：例如 `images/17DRP5sb8fy_r2r_001803`
- `instructions`：文本指令列表（脚本默认取第 1 条）
- `actions`：动作序列（脚本监督的是当前帧对应的下一步动作）

动作映射：

- `0 -> stop`
- `1 -> forward`
- `2 -> left`
- `3 -> right`

## 3. 相关代码功能说明

主脚本：`scripts/train/qwenvl_train/offline_r2r_lora_sft_eval.py`

核心函数：

- `patch_torch_from_numpy()`
  - 针对当前环境中 `torch.from_numpy(np.ndarray)` 异常的兼容补丁。
- `build_samples(data_root, max_samples, seed)`
  - 从离线目录抽取样本，构造成 `(frame_path, instruction, action)`。
- `OfflineR2RSFTDataset`
  - 构造视觉-语言输入，并将 prompt 部分 label 置为 `-100`（只监督答案 token）。
- `QwenVLDataCollator`
  - 对文本序列做 padding，并拼接视觉张量。
- `load_model_with_lora(args)`
  - 加载基座模型并注入 LoRA。
- `run_generation_eval(...)`
  - 用生成方式推断动作，计算 `generation_action_acc`。
- `main()`
  - 串联训练、eval、可选 generation eval，并写入指标文件。

## 4. 训练与评测流程

1. 读取 train/val 的 `annotations.json`，构建样本。
2. 加载 Qwen2.5-VL 模型，注入 LoRA（仅训练 LoRA 参数）。
3. 执行 Trainer 训练，并按 `eval_strategy` 进行验证。
4. 若启用 `--run_generation_eval`，额外执行动作生成准确率评测。
5. 输出模型、checkpoint、以及 `offline_eval_metrics.json`。

## 5. 启动脚本

### 5.1 快速验证（几个 batch / 少量 step）

脚本：`scripts/train/qwenvl_train/run_offline_r2r_quicktest.sh`

用途：快速证明离线训练/测试链路可跑通。

默认行为：

- 小样本（train 64 / val 16）
- `max_steps=5`
- 启用 generation eval

执行：

```bash
bash scripts/train/qwenvl_train/run_offline_r2r_quicktest.sh
```

### 5.2 常规离线训练

脚本：`scripts/train/qwenvl_train/run_offline_r2r_train.sh`

用途：离线训练 + 常规验证 + 可选 generation eval。

执行：

```bash
bash scripts/train/qwenvl_train/run_offline_r2r_train.sh
```

## 6. 关键参数说明

常用参数如下（均可在脚本中改默认值，或改环境变量）：

- `--model_path`
  - 基座模型路径，当前为 `checkpoints/InternVLA-N1-System2`。
- `--train_root` / `--val_root`
  - 离线训练/验证数据目录。
- `--output_dir`
  - 输出目录（模型、checkpoint、指标）。
- `--attn_implementation`
  - 默认 `sdpa`，用于避免 flash-attention 依赖。
- `--dtype`
  - `bf16` / `fp16` / `fp32`。
- `--lora_r` / `--lora_alpha` / `--lora_dropout` / `--lora_target_modules`
  - LoRA 配置。
- `--max_train_samples` / `--max_eval_samples`
  - 样本抽样上限（方便快速试跑）。
- `--max_steps` / `--num_train_epochs`
  - 训练长度控制。
- `--eval_strategy` / `--eval_steps`
  - 验证策略（当前 transformers 版本使用 `eval_strategy`）。
- `--run_generation_eval` / `--generation_eval_samples`
  - 是否计算生成式动作准确率及其样本数。

## 7. 输出与指标

输出目录中常见文件：

- `adapter_model.safetensors`（LoRA 权重）
- `checkpoint-*`（中间 checkpoint）
- `offline_eval_metrics.json`（核心指标）

`offline_eval_metrics.json` 指标说明：

- `train_loss`：训练损失。
- `trainer_eval_*`：Trainer 验证阶段的运行统计。
- `generation_eval_samples`：用于生成评测的样本数。
- `generation_action_acc`：动作准确率。

动作准确率计算方式：

```text
generation_action_acc = correct / total
```

其中 `correct` 表示预测动作与 GT 动作完全一致的样本数。

## 8. 已验证的快速测试结果

已完成一次 5-step 快速验证，输出目录：

- `checkpoints/tmp_offline_lora_quicktest_5steps`

对应指标文件：

- `checkpoints/tmp_offline_lora_quicktest_5steps/offline_eval_metrics.json`

该测试用于证明链路可跑通，不代表最终效果上限。

## 9. 注意事项

- 当前脚本默认走 `sdpa`，不使用 flash-attention。
- 若环境中的 `torch.from_numpy` 异常，脚本会自动启用 fallback 兼容逻辑。
- 少量 step 的 quick test 中 `generation_action_acc` 较低或为 0 属于常见现象。
- 若需要更稳定的指标，建议增大 `max_steps`、`max_train_samples` 与 `generation_eval_samples`。
