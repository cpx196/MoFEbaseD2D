<div align="center">

# MoFEbaseD2D

**GPT-2 Small · FineWeb-Edu 10BT · Factorized Experts**

[中文](#中文) · [English](#english)

<br>

`Dense` &nbsp; `MoFE group LR` &nbsp; `Upcycling` &nbsp; `50k steps`

</div>

<a id="中文"></a>
<details open>
<summary><strong>中文</strong></summary>

## 项目简介

MoFEbaseD2D 研究 GPT-2 Small 在 FineWeb-Edu 10BT 上的稀疏参数扩展，比较三种
相同训练 token 预算的方法：

- **Dense**：原始 GPT-2 Small 继续预训练。
- **MoFE**：在最后三个 GPT-2 MLP block 上使用 Mixture of Factorized Experts。
- **Upcycling**：将最后三个 MLP 替换为完整复制的稀疏专家。

本项目后续所有未特别说明的 `MoFE` 都指 **MoFE group LR**。旧的统一学习率
版本只保留在 [timeline.md](timeline.md) 的历史记录中，不参与最终比较。

## MoFE 方法

MoFE 替换 Transformer block 9、10、11 的 MLP。每层保留一个始终激活的 dense
shared expert，并增加 16 个 factorized private experts。token-choice router 为
每个 token 选择 top-3 private experts。

private experts 使用 `4 x 4` Cartesian factor bank 和独立 expert core。对 expert
`e = 4i + j`，两个投影为：

```text
W1_e = A1_i C1_e B1_j
W2_e = A2_i C2_e B2_j
```

实际执行顺序为 `x -> B -> C -> A`，不会为每个 token 物化完整 expert 矩阵。shared
分支保持原始 GPT-2 MLP，private output core 和 bias 零初始化，使模型在初始化
时严格保持 Dense GPT-2 的函数。

## 最终实验协议

| 配置 | 值 |
| --- | --- |
| 数据 | FineWeb-Edu 10BT，train shards 000-012 |
| Held-out | shard 013 的固定验证集 |
| 硬件 | 4 x RTX 4090 |
| 序列长度 | 1024 |
| Global batch size | 32 sequences |
| 每 step token 数 | 32,768 |
| 训练步数 | 50,000 |
| 每模型训练 token | 1.6384B |
| 计算精度 | BF16 |
| Master 参数 / AdamW states | FP32 |
| Scheduler | Constant，无 warmup |
| Validation | 每 200 step |
| Checkpoint | 每 5,000 step，包含 optimizer/scheduler/RNG state |

MoFE group LR：shared/backbone 为 `1e-5`，private experts 为 `2e-5`，router 为
`3e-5`。Dense 和 Upcycling 使用 `1e-5`。

## 50k 结果

![50k validation loss and token accuracy](results/final_50k/figures/validation_loss_and_token_accuracy_50k.png)

FineWeb-Edu held-out：

| 方法 | Validation loss | PPL | Next-token prediction accuracy |
| --- | ---: | ---: | ---: |
| Dense | 3.112968 | 22.4877 | 40.8511% |
| **MoFE group LR** | **3.092061** | **22.0224** | **41.1613%** |
| Upcycling | 3.104230 | 22.2921 | 40.9910% |

50k 下游任务统一使用原始 `acc`：

| 方法 | LAMBADA acc | HellaSwag acc |
| --- | ---: | ---: |
| Dense | 0.340772 | 0.291675 |
| **MoFE group LR** | **0.343101** | **0.294662** |
| Upcycling | 0.340190 | 0.293467 |

ARC 和 WikiText 不属于最终 benchmark；HellaSwag 的 `acc_norm` 虽存在于原始
lm-eval JSON，但不用于主表。

## 原始实验数据

最终数据集中在 [results/final_50k](results/final_50k/README.md)：

- `validation/validation_loss_50k.csv`：三种方法的 50k validation loss 数据。
- `validation_prediction_accuracy/raw/`：三种方法各 10 个 checkpoint，共 30 个原始 JSON。
- `downstream/`：三种方法 50k step 的 LAMBADA/HellaSwag 原始 JSON。
- `figures/`：最终横向 loss 与 token accuracy 对比图。

旧实验完整保留在 [archive](archive/README.md)，没有删除历史数据。

## 代码与运行

当前实验实现位于 `MoFE/`：

| 路径 | 作用 |
| --- | --- |
| `MoFE/layer.py` | Factorized experts、shared expert、top-k routing |
| `MoFE/modeling.py` | GPT-2 转换和参数统计 |
| `MoFE/train.py` | MoFE group LR 训练入口 |
| `MoFE/train_dense.py` | Dense 训练入口 |
| `MoFE/train_upcycling.py` | Upcycling 训练入口 |
| `MoFE/eval_validation_loss.py` | Held-out loss 评测 |
| `MoFE/eval_validation_token_accuracy.py` | Next-token accuracy 评测 |
| `MoFE/configs/` | E16/K3 配置 |

安装依赖并运行测试：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m unittest \
  MoFE.tests.test_data \
  MoFE.tests.test_mofe \
  MoFE.tests.test_upcycling
```

完整实验时间线见 [timeline.md](timeline.md)。

</details>

<a id="english"></a>
<details>
<summary><strong>English</strong></summary>

## Overview

MoFEbaseD2D studies sparse parameter expansion of GPT-2 Small on FineWeb-Edu
10BT. It compares three methods under the same training-token budget:

- **Dense**: continued pretraining of GPT-2 Small.
- **MoFE**: Mixture of Factorized Experts in the final three GPT-2 MLP blocks.
- **Upcycling**: sparse expansion using complete copies of the final three MLPs.

Unless explicitly labeled otherwise, **MoFE means MoFE group LR**. The older
single-learning-rate MoFE is preserved only as historical context in
[timeline.md](timeline.md) and is excluded from the final comparison.

## Method

MoFE replaces the MLPs in Transformer blocks 9, 10, and 11. Each converted layer
keeps one always-active dense shared expert and adds 16 factorized private
experts. A token-choice router selects the top 3 private experts per token.

With a `4 x 4` Cartesian factor-bank construction and expert `e = 4i + j`:

```text
W1_e = A1_i C1_e B1_j
W2_e = A2_i C2_e B2_j
```

The factorized path runs as `x -> B -> C -> A` without materializing complete
expert matrices per token. The shared branch is copied from the original GPT-2
MLP. Private output cores and biases are zero-initialized, so the converted model
preserves the Dense GPT-2 function at initialization.

## Final Protocol

| Setting | Value |
| --- | --- |
| Data | FineWeb-Edu 10BT, training shards 000-012 |
| Held-out set | Fixed validation tail from shard 013 |
| Hardware | 4 x RTX 4090 |
| Sequence length | 1024 |
| Global batch size | 32 sequences |
| Tokens per step | 32,768 |
| Optimizer steps | 50,000 |
| Training tokens per model | 1.6384B |
| Compute | BF16 |
| Master parameters / AdamW states | FP32 |
| Scheduler | Constant, no warmup |
| Validation | Every 200 steps |
| Checkpoint | Every 5,000 steps, including optimizer/scheduler/RNG state |

MoFE group LR uses `1e-5` for the backbone/shared experts, `2e-5` for private
experts, and `3e-5` for routers. Dense and Upcycling use `1e-5`.

## Results at 50k

![50k validation loss and token accuracy](results/final_50k/figures/validation_loss_and_token_accuracy_50k.png)

FineWeb-Edu held-out results:

| Method | Validation loss | PPL | Next-token prediction accuracy |
| --- | ---: | ---: | ---: |
| Dense | 3.112968 | 22.4877 | 40.8511% |
| **MoFE group LR** | **3.092061** | **22.0224** | **41.1613%** |
| Upcycling | 3.104230 | 22.2921 | 40.9910% |

Downstream results use the original `acc` metric for both tasks:

| Method | LAMBADA acc | HellaSwag acc |
| --- | ---: | ---: |
| Dense | 0.340772 | 0.291675 |
| **MoFE group LR** | **0.343101** | **0.294662** |
| Upcycling | 0.340190 | 0.293467 |

ARC and WikiText are excluded from the final benchmark. HellaSwag `acc_norm` is
present in raw lm-eval JSON but is not used in the primary table.

## Raw Experiment Data

The final archive is in [results/final_50k](results/final_50k/README.md):

- `validation/validation_loss_50k.csv`: 50k held-out loss data for all three methods.
- `validation_prediction_accuracy/raw/`: 30 original JSON points, 10 checkpoints per method.
- `downstream/`: original 50k LAMBADA/HellaSwag JSON outputs for all three methods.
- `figures/`: the final side-by-side loss and token-accuracy figure.

All historical experiments remain available in [archive](archive/README.md).

## Code and Usage

The active implementation lives in `MoFE/`. Install dependencies and run tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m unittest \
  MoFE.tests.test_data \
  MoFE.tests.test_mofe \
  MoFE.tests.test_upcycling
```

The full experiment record is maintained in [timeline.md](timeline.md).

## Upstream

This project is an independent derivative of
[D2DMoE](https://github.com/bartwojcik/D2DMoE), based on upstream commit
`a7027cdc1f01c9c618c39eebe639d1664549b066`. The upstream project and this
derivative use the MIT License. The associated paper is:

> Filip Szatkowski, Bartosz Wojcik, Mikolaj Piorczynski, Simone Scardapane.
> Exploiting Activation Sparsity with Dense to Dynamic-k Mixture-of-Experts
> Conversion. NeurIPS 2024. <https://arxiv.org/abs/2310.04361>

</details>
