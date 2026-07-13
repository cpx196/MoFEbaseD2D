# MoFEbaseD2D

本仓库基于 [D2DMoE](https://github.com/bartwojcik/D2DMoE) 开展 GPT-2
实验，目前包含两部分工作：可复现的 GPT-2 Small dense 基线，以及在其最后
3 个 Transformer block 上构建的 MoFE（Mixture of Factorized Experts）方法。

当前公开仓库：<https://github.com/cpx196/MoFEbaseD2D>

## 时间线

### 2026-07-13：建立 GPT-2 Small dense 基线

- 从上游 D2DMoE commit `a7027cdc` 开始整理实验环境。
- 创建 `d2d-gpt` Conda 环境，使用 Python 3.11、PyTorch 2.5.1+cu124。
- 加载 `openai-community/gpt2`，确认模型为 GPT-2 Small、12 层、
  `124,439,808` 参数。
- 在单张 RTX 4090 上完成固定 prompt 推理和 WikiText-2 validation
  perplexity 计算。
- 使用 `lm-evaluation-harness` 完成六项 zero-shot 下游评测。

Dense 基线结果：

| 任务 | 指标 | 分数 |
| --- | --- | ---: |
| WikiText-2 validation | perplexity | 26.6188 |
| LAMBADA OpenAI | accuracy | 0.3097 |
| HellaSwag | normalized accuracy | 0.3122 |
| PIQA | normalized accuracy | 0.6257 |
| WinoGrande | accuracy | 0.5138 |
| ARC Easy | normalized accuracy | 0.3977 |
| ARC Challenge | normalized accuracy | 0.2287 |

原始汇总保存在
[metrics_summary.csv](results/dense_gpt2_small/metrics_summary.csv)，结果图位于
`results/dense_gpt2_small/figures/`。

![GPT-2 Small dense baseline](results/dense_gpt2_small/figures/baseline_overview.png)

### 2026-07-13：整理为独立公开仓库

- 保留上游 MIT License、论文引用和 `upstream` Git remote。
- 增加缓存、数据集、权重、日志和密钥的忽略规则。
- 增加经过验证的 dense 评测依赖和三个基线评测/绘图脚本。
- 将 dense 汇总 CSV 和结果图纳入版本管理，不提交模型权重和数据集。
- 仓库整理提交：`50c8b67`。

### 2026-07-13：定义 MoFE 结构

根据 `D2Dinstr/MoFE_GPT2_MoFEversion.md` 确定主实验结构：

- 只替换 GPT-2 block 9、10、11 的 MLP。
- 每层保留一个始终激活的 dense shared expert。
- 每层增加 16 个 factorized private experts。
- 4 组 A 和 4 组 B 通过笛卡尔积组合，每个 private expert 使用独立 core。
- Router 对每个 token 从 16 个 private experts 中选择 top-3。
- Shared expert 不参与路由，也不由 router 权重缩放。

Private expert 的两个投影均按以下形式构建：

```text
W1_e = A1_i C1_e B1_j
W2_e = A2_i C2_e B2_j
e = 4i + j
```

GPT-2 hidden size 为 768，主配置 rank 为 576。A/B 使用 dense FFN 权重的
行列切片初始化，core 使用标准差 0.025 的独立随机初始化。

### 2026-07-13：完成 MoFE 代码与验证链路

- 在 `MoFE/` 中实现模型转换、factorized expert、token-choice top-3 路由、
  balance loss 和 router z-loss。
- Private expert 按 `x -> B -> core -> A` 执行，不为每个 token 物化完整权重。
- 增加 MoFE 配置、参数分类统计、checkpoint 保存/重载和初始化报告。
- 增加支持 Accelerate/DDP 的训练入口，初步训练预算为 200 optimizer steps。
- 训练数据集没有写死；启动时必须显式传入 `--dataset-name` 或
  `--train-file`。
- 增加跨 rank expert 负载、router entropy、tokens/s、峰值显存和
  private/shared 输出范数监控。
- MoFE 实现提交：`06ea609`。

真实 GPT-2 Small 构建结果：

| 项目 | 结果 |
| --- | ---: |
| 总参数量 | 209,595,696 |
| Shared expert 参数 | 14,167,296 |
| A/B factor bank 参数 | 53,084,160 |
| Core 参数 | 31,850,496 |
| Private bias 参数 | 184,320 |
| Router 参数 | 36,912 |

只启用 shared 分支时，MoFE 与原 dense GPT-2 的 logits 最大误差为 `0.0`。
随机 private 分支直接以完整强度启用时，一个短 prompt 上观测到的最大 logit
差为 `126.83`。该数值不是准确率，而是初始化扰动指标，因此训练入口默认让
private scale 在 200 steps 内从 0 线性增加到 1。

完成的验证包括：

- 5 项离线单元测试：shared 权重、A/B 切片、因子计算、top-3 路由与梯度、
  checkpoint 重载。
- 真实 GPT-2 Small 的三层 MoFE 构建和短序列前向。
- Tiny GPT-2 的 2-step 端到端训练、日志和 final checkpoint 重载。

### 2026-07-13：评测未 warmup 的初始化 MoFE

为了直接验证随机 private 分支全开造成的影响，构建 MoFE 后不执行训练，固定
`private_scale=1.0`，直接运行与 dense 基线相同的六项 zero-shot 任务。该实验
是 initialization-only 对照，不代表训练后的 MoFE。

| 任务 | Dense | MoFE no-warmup | 绝对变化 |
| --- | ---: | ---: | ---: |
| LAMBADA OpenAI | 0.3097 | 0.0000 | -0.3097 |
| HellaSwag | 0.3122 | 0.2619 | -0.0503 |
| PIQA | 0.6257 | 0.5114 | -0.1143 |
| WinoGrande | 0.5138 | 0.5036 | -0.0103 |
| ARC Easy | 0.3977 | 0.2546 | -0.1431 |
| ARC Challenge | 0.2287 | 0.2637 | +0.0350 |

![Dense vs MoFE without warmup](results/mofe_gpt2_no_warmup/figures/no_warmup_vs_dense.png)

除 ARC Challenge 外，其余五项均下降；更重要的是，MoFE 分数整体接近各
选择题任务的随机猜测水平，LAMBADA accuracy 降为 0。ARC Challenge 的
`0.2637` 也接近四选一随机水平，不能解释为能力提升。这与初始化时观测到的
巨大 logit 扰动一致，说明不能跳过 private 分支 warmup 或等效的稳定初始化。

比较数据保存在
[`no_warmup_vs_dense.csv`](results/mofe_gpt2_no_warmup/figures/no_warmup_vs_dense.csv)。
完整 `lm-eval` JSON 和日志保存在仓库外的数据目录
`/home/pxchen/data/pxchen/results/mofe_gpt2_no_warmup/`。

## 当前状态

已经完成 dense 基线、MoFE 架构代码、初始化验证、训练链路验证，以及未训练、
未 warmup 初始化模型的下游对照。尚未选择 MoFE continued-training 数据集，
因此正式 200-step 训练尚未运行；当前结果只能说明随机 private 分支直接全开会
破坏原模型能力，不能用于判断训练后 MoFE 的最终性能。

下一阶段顺序为：

1. 确定 continued-training 数据集和数据预处理。
2. 生成真实模型 `initialization_report.json`。
3. 运行 200-step 工程训练，检查 loss、负载、范数、显存和吞吐。
4. 决定是否扩大训练预算，并与 dense、dense upcycling MoE 做公平对比。
5. 使用与 dense 基线相同的 WikiText-2 和六项 zero-shot 协议评测。

## 主要文件

| 路径 | 作用 |
| --- | --- |
| `MoFE/layer.py` | Shared expert、factorized private experts 和路由前向 |
| `MoFE/modeling.py` | GPT-2 block 替换、损失、统计和参数量 |
| `MoFE/train.py` | 默认 200-step 的通用训练入口 |
| `MoFE/validate_initialization.py` | 真实 GPT-2 初始化报告 |
| `MoFE/eval_no_warmup.py` | 未训练、无 warmup MoFE 的 lm-eval 入口 |
| `MoFE/plot_no_warmup_comparison.py` | Dense/MoFE 对比 CSV 与柱状图 |
| `MoFE/tests/test_mofe.py` | 无数据集依赖的单元测试 |
| `MoFE/configs/` | MoFE 主实验配置 |
| `MoFE_METHOD.md` | MoFE 数学定义、初始化和训练细节 |
| `moe_block/` | 构建 MoFE 时参考的原始实现 |
| `D2Dinstr/` | Dense 基线和 MoFE 实验任务说明 |

运行离线测试：

```bash
python -m unittest MoFE.tests.test_mofe
```

训练命令将在数据集确定后补充到本时间线。
