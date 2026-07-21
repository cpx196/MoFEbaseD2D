# MoFEbaseD2D Timeline

This file preserves the chronological development and experiment record that
previously lived in `README.md`. Historical entries use the model names that
were current when each experiment ran. From the final 50k comparison onward,
unqualified **MoFE** means **MoFE group LR** (`1e-5` shared/backbone, `2e-5`
private, `3e-5` router); the older single-learning-rate model is called
**MoFE baseline** and is no longer used for new experiments.

> Historical artifacts remain in the repository and may additionally refer to
> the external runtime archive. The compact final 50k raw-data archive is under
> `results/final_50k/`.

本仓库基于 [D2DMoE](https://github.com/bartwojcik/D2DMoE) 开展 GPT-2
实验，目前包含三部分工作：可复现的 GPT-2 Small dense 基线、在其最后
3 个 Transformer block 上构建的 MoFE（Mixture of Factorized Experts），以及
将最后 3 个 FFN 替换为 16 个完整复制专家的 Sparse Upcycling 对照。

当前公开仓库：<https://github.com/cpx196/MoFEbaseD2D>

项目中期曾使用 handoff 文档记录跨会话状态；项目收尾时其中仍有效的内容已合并
到本时间线和最终 README，旧 handoff 文件已退休。

## 时间线

### 2026-07-13

#### 建立 GPT-2 Small dense 基线

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

该阶段的原始汇总和结果图属于早期 pilot，仍保留在仓库的历史 `results/` 目录中。

#### 整理为独立公开仓库

- 保留上游 MIT License、论文引用和 `upstream` Git remote。
- 增加缓存、数据集、权重、日志和密钥的忽略规则。
- 增加经过验证的 dense 评测依赖和三个基线评测/绘图脚本。
- 将 dense 汇总 CSV 和结果图纳入版本管理，不提交模型权重和数据集。
- 仓库整理提交：`50c8b67`。

#### 定义 MoFE 结构

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

#### 完成 MoFE 代码与验证链路

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

#### 评测未 warmup 的初始化 MoFE

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

除 ARC Challenge 外，其余五项均下降；更重要的是，MoFE 分数整体接近各
选择题任务的随机猜测水平，LAMBADA accuracy 降为 0。ARC Challenge 的
`0.2637` 也接近四选一随机水平，不能解释为能力提升。这与初始化时观测到的
巨大 logit 扰动一致，说明不能跳过 private 分支 warmup 或等效的稳定初始化。

该阶段的仓库内比较 CSV 和图片仍保留为历史参考。完整 `lm-eval` JSON 和日志
曾保存在仓库外的数据目录
`/home/pxchen/data/pxchen/results/mofe_gpt2_no_warmup/`。

### 2026-07-14

#### 定位首次 WikiText-103 100-step 训练异常

- 选择 WikiText-103 train 作为 MoFE continued-training 数据，使用 GPT-2
  tokenizer 打包为 1024-token 序列。
- 首次 4 卡运行中，LM loss 从 `3.7386` 上升至 `12.5940`，未满足短程训练
  loss 下降的验收条件；该失败 checkpoint 已删除。
- 使用原始 dense GPT-2 在相同 packed 数据随机抽取 16 条序列，平均 loss 为
  `3.5247`，确认数据和 causal-LM preprocessing 不是 loss 爆炸的主因。
- 初始化 private/shared 输出范数比在 block 9、10、11 分别约为
  `13.2、14.2、29.0`。线性 private warmup 到 step 5 时，最后一层 private
  分支的实际强度已经超过 shared 分支。
- 同时发现 Accelerate 在 4 卡下将 scheduler 推进了 4 倍：optimizer 为
  100 steps，而 scheduler 到达 400。该问题导致 cosine LR 多次降至 0 后回升。

#### 修复四卡训练链路与实时监控

- 关闭 Accelerate 按进程数自动推进 scheduler，只在真实 optimizer update 后
  调用一次 `scheduler.step()`。复测 checkpoint 中 optimizer 和 scheduler 均为
  100 steps。
- 将每卡 batch size 从 1 提高到 4，同时将 gradient accumulation 从 8 降到
  2；四卡有效 batch 仍为 32 条序列，即每个 optimizer step 处理 32,768 tokens。
- 增加启动配置摘要、clipping 前 gradient norm、紧凑单行 step 日志和强制
  flush。完整 routing 统计继续保存为 JSONL。
- batch 4 运行峰值 PyTorch allocated memory 为 `10.59 GiB/卡`，最终吞吐为
  `63.1k tokens/s`；相较 batch 1 的约 `23.9k tokens/s` 提高约 2.6 倍。

#### 使用保持原函数的 private 初始化

原始实现同时随机初始化 `C1` 和 `C2`，导致未经训练的 private 分支产生巨大
输出。稳定版本保留 A/B dense 切片和随机输入 core，但将输出 core 初始化为零：

```text
C1_e ~ Normal(0, 0.025^2)
C2_e = 0
```

因此初始化时每个 private expert 输出严格为 0，完整 MoFE 与 dense GPT-2 的
logits 最大误差为 `0.0`。训练开始后 `C2` 先获得梯度；随着 `C2` 离开零点，
A/B 和 `C1` 逐步获得语言模型梯度。该变化只修改初始化，不改变 token-choice
top-3 路由和最终 MoFE 表达能力。

离线单元测试覆盖了 `C2` 零初始化、完整 MoFE/dense logits 对齐、`C2` 非零
梯度、factorized/materialized 前向一致性和 checkpoint 重载。真实 GPT-2 初始化
报告属于早期工程验证，并保留为历史参考，不纳入最终 50k 对比。

#### 完成稳定的 100-step 工程训练

使用 4 张 RTX 4090、bf16、每卡 batch 4、gradient accumulation 2、LR
`1e-5`、10-step LR warmup 和 100-step private scale warmup 完成 WikiText-103
短程训练。

| 指标 | Step 1 | 最低记录值 | Step 100 |
| --- | ---: | ---: | ---: |
| LM loss | 3.6908 | 3.2503 | 3.3153 |
| Total loss | 3.7088 | 3.2684 | 3.3333 |
| Pre-clip gradient norm | 6.1875 | 3.0625 | 6.4375 |

Step 100 的 block 9、10、11 raw private/shared 输出范数比分别为
`0.96、0.47、0.61`，不再出现随机 private 分支压倒 shared 分支的情况。三个
输出 core 的 norm 分别增长到 `0.346、0.320、0.303`，确认 private 分支已从零
初始化中启动。这里的 loss 是工程训练日志中的采样 microbatch loss，不替代独立
validation loss。

训练 JSONL 仍保留为历史记录；
[作图脚本](MoFE/plot_training_loss.py) 可从同格式日志重新生成结果图。

### 2026-07-15

#### 建立匹配的 1000-step 固定验证对照

- 修复训练日志只记录 rank 0 最后一个 microbatch 的问题。当前训练 loss 是两卡、
  4 次梯度累积组成的完整 effective batch 32 平均值。
- 固定 WikiText-103 validation 为 240 条 1024-token 序列，共 245,520 个预测
  token；Dense、MoFE 和 Upcycling 使用相同 validation、数据顺序、effective
  batch、学习率与 optimizer steps。
- 三个实验均使用 10-step LR warmup，之后保持常数学习率 `1e-5`。

Step-1000 固定验证结果：

| 模型 | 参数量 | Validation loss | Perplexity |
| --- | ---: | ---: | ---: |
| Dense GPT-2 | 124,439,808 | 3.315577 | 27.5383 |
| MoFE E16 K3 | 209,595,696 | **3.113023** | **22.4889** |
| Sparse Upcycling E16 K3 | 336,986,160 | 3.316069 | 27.5518 |

每个训练日志包含 201 个 effective-batch 采样点（step 1，之后每 5 steps）。
浅色线为原始采样，粗线为 50-step moving average。该早期图表保留为历史记录，
不纳入最终归档。

#### 完成 MoFE private 分支消融

在同一个 step-1000 checkpoint 上只将 private output scale 从 1 设为 0，shared
分支和 GPT-2 其余主干保持不变：

| 配置 | WT103 fixed loss | WT103 PPL | WT2 PPL |
| --- | ---: | ---: | ---: |
| Private OFF | 3.499410 | 33.0959 | 29.0748 |
| Private ON | **3.112870** | **22.4855** | **20.8666** |

Private ON 相对 OFF 将 WT103 PPL 降低 32.06%，将 WT2 PPL 降低 28.23%。该
差值证明 private 输出有预测贡献，但不能直接解释为参数贡献百分比。

#### 增加 Sparse Upcycling 对照

- 将 block 9、10、11 的原 FFN 分别替换为 16 个完整参数副本，不保留额外
  shared FFN。
- Token-choice router 选择 top-3，top-k 权重归一化后聚合；step-0 固定验证
  loss 为 3.430742，与 Dense 起点保持一致。
- 双卡、每卡 batch 4、gradient accumulation 4，训练 1000 optimizer steps；
  吞吐约 39.1k tokens/s，PyTorch peak allocated memory 为 11.09 GiB/卡。
- `step_001000` 保存模型、AdamW、scheduler 和两卡 RNG state，可继续训练；
  `final` 保存独立评测权重。

当前 1000-step 预算下 Upcycling 与 Dense 基本持平。训练后完整复制专家之间的
相对参数分化仅约 0.11%--0.15%，说明额外容量尚未充分展开；该现象需要更长
continued pretraining、更多样的数据和进一步 router 消融验证。

### 2026-07-16

#### 整理离线打包目录

- GitHub HTTPS push 因当前网络环境的 GnuTLS handshake 失败，改为直接打包
  下载源码仓库。
- checkpoint、数据集、Hugging Face cache、训练日志和完整评测 artifacts 均未
  删除，已整体移动到仓库同级目录
  `/home/iot-mengshiyuan/MoFEbaseD2D_data/`。
- 仓库内只保留约 1 MB 的关键图表、CSV、JSON 和 Markdown 汇总，源码仓库
  从约 28 GB 降至约 4.3 MB。
- 清理 `__pycache__` 等可再生成文件；上游 D2DMoE 源码只有约 2 MB，继续保留
  以维持来源、许可和实现参考，不为节省极少空间破坏代码结构。

## 当前状态

目前已完成：

1. Dense、MoFE、Sparse Upcycling 三种模型的匹配 1000-step 训练。
2. 跨卡、跨梯度累积的正确训练 loss 聚合，以及固定 validation 评测链路。
3. MoFE step-3000 continued-training checkpoint 与既有 zero-shot 评测。
4. MoFE step-1000 private ON/OFF 消融和 WikiText-2 PPL 对照。
5. Upcycling 模型转换、checkpoint、可续训状态、单元测试和结果曲线。

当前有效结论来自 FP32 master 参数与 FP32 AdamW states 的 200-step 修正实验：
Dense 和 MoFE 都能正常下降，MoFE 在相同 6.55M token 后的 held-out PPL 比 Dense
低 1.06%。此前 1000-step 实验将参数和 Adam moments 都存为 BF16，对非零预训练
权重造成严重更新量化，相关 4.45% 优势只作为历史 pilot 保留，不再作为方法结论。
下一阶段继续使用独立 FineWeb-Edu held-out，并增加等 FLOPs、router shuffle 和
多随机种子实验。历史 WikiText 结果同样不再用于训练验证或 checkpoint 选择。

## 数据与 Checkpoint

大型文件不包含在离线源码包中。服务器当前布局为：

```text
/home/iot-mengshiyuan/
├── MoFEbaseD2D/          # 源码、README、关键小型结果
└── MoFEbaseD2D_data/     # 28 GB 数据、cache、checkpoint、完整日志
    └── pxchen/
        ├── checkpoints/
        ├── datasets/
        ├── hf/
        ├── hf_models/
        ├── results/
        └── runs/
```

在服务器继续实验时建议设置：

```bash
export DATA_ROOT=/home/iot-mengshiyuan/MoFEbaseD2D_data/pxchen
export HF_HOME="$DATA_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

项目收尾后将 `results/final_50k/` 作为最终小型数据归档，同时保留历史结果文件。
checkpoint、完整日志和数据集继续位于外部 runtime，下载到其他机器时需要单独复制。

## 主要文件

| 路径 | 作用 |
| --- | --- |
| `MoFE/layer.py` | Shared expert、factorized private experts 和路由前向 |
| `MoFE/modeling.py` | GPT-2 block 替换、损失、统计和参数量 |
| `MoFE/train.py` | 默认 200-step 的通用训练入口 |
| `MoFE/train_dense.py` | 匹配超参数的 Dense 训练入口 |
| `MoFE/upcycling.py` | 完整 FFN 复制的 Sparse Upcycling 层和 checkpoint |
| `MoFE/train_upcycling.py` | Sparse Upcycling 分布式训练入口 |
| `MoFE/validate_initialization.py` | 真实 GPT-2 初始化报告 |
| `MoFE/eval_no_warmup.py` | 未训练、无 warmup MoFE 的 lm-eval 入口 |
| `MoFE/plot_no_warmup_comparison.py` | Dense/MoFE 对比 CSV 与柱状图 |
| `MoFE/plot_training_loss.py` | 从训练 JSONL 生成 loss 曲线 |
| `MoFE/tests/test_mofe.py` | 无数据集依赖的单元测试 |
| `MoFE/configs/` | MoFE 主实验配置 |
| `README.md` | 最终 MoFE 数学定义、初始化、训练协议和结果 |
| `moe_block/` | 构建 MoFE 时参考的原始实现 |
| `D2Dinstr/` | Dense 基线和 MoFE 实验任务说明 |

运行离线测试（当前共 13 项）：

```bash
python -m unittest \
  MoFE.tests.test_data MoFE.tests.test_mofe MoFE.tests.test_upcycling
```

本机 FineWeb-Edu `sample/10BT` 位于
`/data/chenpengxu/HMoE_data/fineweb_10BT`，包含 14 个 Parquet 分片、
9,672,101 篇文档和约 9.97B tokens。三个训练入口均可将该目录直接传给
`--train-file`；Parquet 目录会按文件名排序并自动启用 streaming，不会生成全量
tokenized 中间缓存。

以下命令已在四张 RTX 4090 上完成 2-step MoFE smoke；它只验证数据和分布式
训练链路，不代表正式长训练配置：

```bash
export HF_HOME=/data/chenpengxu/MoFEbaseD2D_runtime/hf

accelerate launch --multi_gpu --num_processes 4 -m MoFE.train \
  --train-file /data/chenpengxu/HMoE_data/fineweb_10BT \
  --model-name-or-path openai-community/gpt2 \
  --output-dir /data/chenpengxu/MoFEbaseD2D_runtime/runs/mofe_fineweb_smoke \
  --block-size 1024 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-steps 2 \
  --learning-rate 5e-5 \
  --warmup-steps 0 \
  --scheduler constant \
  --private-warmup-steps 0 \
  --logging-steps 1 \
  --num-workers 0 \
  --shuffle-buffer-size 256 \
  --preprocessing-batch-size 16
```

#### FineWeb-Edu 四卡 1000-step 结果

以下 1000-step 结果使用 BF16 master 参数和 BF16 Adam moments，存在架构相关的
更新量化偏差，只作为历史记录。使用每卡 batch 4、gradient accumulation 2、effective batch 32、10-step LR
warmup 后恒定 `1e-5`，以及 100-step private scale warmup 完成正式训练。为控制
主机内存，streaming 使用 2,048 文档 shuffle buffer 和 0 个 DataLoader worker；
训练入口现会对 streaming 数据自动强制该 worker 设置。

| 数据与配置 | GPT-2 起点 | Final private OFF | Final private ON |
| --- | ---: | ---: | ---: |
| FineWeb-Edu held-out loss | 3.299973 | 3.300422 | **3.235094** |
| FineWeb-Edu held-out PPL | 27.1119 | 27.1241 | **25.4088** |

FineWeb-Edu held-out PPL 相比原始 GPT-2 降低 6.28%，且 private OFF 基本回到
起点，说明域内增益主要来自 private 分支。最终吞吐约 43.7k tokens/s，PyTorch
peak allocated memory 为 10.58 GiB/卡。完整 checkpoint、训练日志和评测位于：

```text
/data/chenpengxu/MoFEbaseD2D_runtime/runs/
  mofe_gpt2_finewebedu10bt_4gpu_1000step_b4_ga2_fixedval_sb2048_nw0_20260716
```

对 step 100 至 1000 的十个 checkpoint 使用同一 FineWeb-Edu held-out 重新评测后，
validation loss 从 step 100 的 `3.264837` 连续下降到 step 900 的 `3.234236`；
step 1000 轻微回升 `0.000859` 至 `3.235094`。当前最佳保存点是 step 900，完整
曲线和精确表格位于上述运行目录的 `reports/fineweb_validation_curve`。

使用完全匹配的数据流、seed、序列长度、effective batch、学习率和 1000 steps
完成四卡 Dense GPT-2 对照。统一 checkpoint 重放精度后：

| 模型 | Step 0 loss | Step 1000 loss | Step 1000 PPL | Tokens/s | Peak GiB/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 3.299059 | 3.280629 | 26.5925 | 83,297 | 7.46 |
| MoFE | 3.299059 | **3.235094** | **25.4088** | 43,658 | 10.58 |

在该历史 BF16 设置下，Dense PPL 相对起点降低 1.83%，MoFE step 1000 PPL 比
Dense 低 4.45%；该差值已被 FP32-master 对照证明明显高估。Dense 和 MoFE 的
最佳已保存点均为 step 900。输出及完整对比图表位于：

```text
/data/chenpengxu/MoFEbaseD2D_runtime/runs/
  dense_gpt2_finewebedu10bt_4gpu_1000step_b4_ga2_finewebval_sb2048_nw0_20260717
/data/chenpengxu/MoFEbaseD2D_runtime/comparisons/
  dense_mofe_finewebedu_4gpu_1000step_20260717
```

使用固定 FineWeb-Edu validation 选出的双方 step-900 checkpoint，完成四项
lm-eval 0.4.12 zero-shot 下游评测：

| 任务 | 指标 | Dense | MoFE | MoFE - Dense |
| --- | --- | ---: | ---: | ---: |
| LAMBADA OpenAI | acc | 0.308558 | 0.305647 | -0.002911 |
| HellaSwag | acc_norm | 0.311591 | 0.311790 | +0.000199 |
| PIQA | acc_norm | 0.626224 | 0.623504 | -0.002720 |
| WinoGrande | acc | 0.505919 | 0.511444 | +0.005525 |

Dense 与 MoFE 各胜两项，四项宏平均分别为 `0.438073` 和 `0.438096`，差值只有
`+0.000023`，且所有 accuracy 差值均小于各自标准误，当前不能判定明确的下游
赢家。MoFE 的 LAMBADA PPL 低 0.86%，但 exact-match accuracy 略低。原始 JSON、
CSV、图和报告位于上述 comparison 目录的 `downstream` 子目录。

使用 CalFLOPs 0.3.2 对相同 batch 和序列长度做理论计算量统计。profiling 时使用
eager attention，以便 CalFLOPs 能观察到 fused SDPA 隐藏的 QK^T 和 AV 矩阵乘法。
在训练使用的序列长度 1024 下，Dense 和 MoFE 每个 1024-token 序列的前向计算量
分别为 `291.898 GFLOPs` 和 `385.741 GFLOPs`，MoFE 是 Dense 的 `1.3215x`。
按 backward = 2x forward、每步 32768 token 估算，每步训练计算量分别为
`28.022 TFLOPs` 和 `37.031 TFLOPs`。因此当前 1000/1000 step 是等 token 比较；
等计算量时，1000 MoFE step 对应约 `1321.5` Dense step，而 1000 Dense step 对应
约 `756.7` MoFE step。完整结果位于 comparison 目录的
`compute/calflops_profile.json`，复现入口为 `python -m MoFE.profile_flops`。

### FP32 master、固定学习率、无 warmup 的 200-step 对照

Dense 和 MoFE 使用文件级隔离的 FineWeb-Edu 训练分片 000--012，并统一在 shard
013 的固定 held-out 上每 10 step 做一次 validation。双方均使用 4 卡、序列长度
1024、effective batch 32、固定学习率 `1e-5`、0 LR warmup 和 200 optimizer steps；
MoFE 的 private scale 从 step 0 起固定为 1，不做 private warmup。每个模型共看到
`6,553,600` 个训练 token。模型 master 参数与 AdamW 一阶、二阶矩保持 FP32，
仅前向和反向计算使用 BF16 autocast。

| 模型 | Step 0 loss | Step 200 loss | Loss 变化 | Step 200 PPL |
| --- | ---: | ---: | ---: | ---: |
| Dense | 3.299973 | 3.221984 | -0.077989 | 25.0778 |
| MoFE | 3.299973 | **3.211319** | **-0.088654** | **24.8118** |

step 200 时 MoFE loss 比 Dense 低 `0.010665`，PPL 低 `1.06%`；双方最佳验证点
均为 step 200。Dense 吞吐为 61,318 token/s，MoFE 为 32,434 token/s，峰值显存
分别为 8.85 和 13.25 GiB/GPU。旧 BF16-master 版本只有 2.99% 的 Dense 参数发生
可表示变化，修正后为 99.9986%，所以旧 200-step 的 4.38% PPL 差距无效。21 个
validation 点、10-step 训练 loss 采样、图和报告位于：

```text
/data/chenpengxu/MoFEbaseD2D_runtime/comparisons/
  dense_mofe_finewebedu_4gpu_200step_fp32master_constlr_nowarmup_val10_20260717
```

双方 step-200 checkpoint 还使用 lm-eval 0.4.12 完成四项 zero-shot 下游评测：

| 任务 | Dense | MoFE | MoFE - Dense |
| --- | ---: | ---: | ---: |
| LAMBADA OpenAI acc | 0.320396 | 0.319814 | -0.000582 |
| HellaSwag acc_norm | 0.313483 | 0.312189 | -0.001295 |
| PIQA acc_norm | 0.630033 | 0.621328 | -0.008705 |
| WinoGrande acc | 0.513812 | 0.501973 | -0.011839 |

四任务宏平均为 Dense `0.444431`、MoFE `0.438826`，MoFE 低 `0.005605`。Dense
四项数值都更高，但所有单项差异均小于报告的独立标准误，当前不能判定显著的下游
赢家。LAMBADA PPL 为 Dense `36.0339`、MoFE `36.6882`，MoFE 高 1.82%。原始
JSON、CSV、图和报告位于上述 comparison 目录的 `downstream` 子目录。

### Sparse Upcycling FP32-master 200-step 对照

遗漏的 Sparse Upcycling E16 K3 也使用相同协议重跑：最后 3 个 FFN 各替换为
16 个完整复制专家，token-choice top-3；FP32 master 参数与 AdamW states、BF16
计算、固定 `1e-5`、无 warmup、effective batch 32、6,553,600 token，并每 10 step
在同一 FineWeb-Edu held-out 上验证。

| 模型 | 参数量 | Step 200 loss | Step 200 PPL | Tokens/s | Peak GiB/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 124,439,808 | 3.221984 | 25.0778 | 61,318 | 8.85 |
| MoFE | 209,595,696 | **3.211319** | **24.8118** | 32,434 | 13.25 |
| Upcycling | 336,986,160 | 3.225175 | 25.1580 | 29,053 | 14.88 |

Upcycling final loss 比 Dense 高 `0.003191`，比 MoFE 高 `0.013856`。所有 16 个专家
都被使用，但 step-200 router entropy 为 `2.7665--2.7670`，接近
`log(16)=2.7726`；各层专家相对 expert 0 的平均参数分歧只有 `0.17%--0.21%`。
完整复制专家在短预算内仍未形成强分化。三模型日志、CSV、曲线和报告位于：

```text
/data/chenpengxu/MoFEbaseD2D_runtime/comparisons/
  dense_mofe_upcycling_finewebedu_4gpu_200step_fp32master_constlr_nowarmup_val10_20260717
```

从 2026-07-17 起，训练验证、checkpoint 选择和方法比较统一使用固定且与训练
数据不重叠的 FineWeb-Edu held-out。WikiText 相关结果只作为历史记录保留，不再
继续使用或扩展。

### FineWeb-Edu GBS32 50k 主实验与三方法对照

当前主实验统一使用 FineWeb-Edu 10BT、固定 held-out、4 张 RTX 4090、序列长度
1024、effective batch 32、固定学习率 `1e-5`、0 LR warmup、FP32 master 参数与
FP32 AdamW states，BF16 计算。每个 optimizer step 处理 32,768 tokens，50k step
对应 1.6384B training tokens。训练 loss 每 50 step 记录一次，validation 每 200
step 记录一次，checkpoint 每 5k step 保存一次，并包含 optimizer、scheduler 与
random states 以支持后续续训。

三种方法参数量如下：

| 方法 | 总参数量 | 可训练参数量 | 相对 Dense |
| --- | ---: | ---: | ---: |
| Dense GPT-2 Small | 124,439,808 | 124,439,808 | 1.00x |
| MoFE E16 K3 | 209,595,696 | 209,595,696 | 1.68x |
| Sparse Upcycling E16 K3 | 336,986,160 | 336,986,160 | 2.71x |

MoFE 参数细分为 remaining backbone `110,272,512`、shared experts `14,167,296`、
factor banks `53,084,160`、cores `31,850,496`、private biases `184,320` 和
routers `36,912`。Sparse Upcycling 参数细分为 remaining backbone `110,272,512`、
完整复制 experts `226,676,736` 和 routers `36,912`。因此 Upcycling 比 MoFE 多
约 127.39M 参数。

Dense、MoFE baseline 和 Sparse Upcycling 均已完成 50k step：

| 模型 | Final step | Final validation loss | Final PPL | Best step | Best validation loss | Best PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 50,000 | 3.112968 | 22.4877 | 49,600 | 3.112472 | 22.4765 |
| MoFE baseline | 50,000 | **3.097295** | **22.1380** | 49,600 | **3.096505** | **22.1205** |
| Sparse Upcycling | 50,000 | 3.104230 | 22.2921 | 49,600 | 3.103765 | 22.2817 |

在 50k step 上，MoFE baseline 的 held-out validation loss 比 Dense 低
`0.015673`，比 Sparse Upcycling 低 `0.006935`。该历史排序为 MoFE baseline <
Upcycling < Dense。完整
validation 曲线、训练 loss 曲线和图表位于仓库外 runtime 归档目录：

```text
/data/chenpengxu/MoFEbaseD2D_runtime/by_date/2026-07-20/comparisons/
  gbs32_three_model_50k_downstream_validation_20260720
/data/chenpengxu/MoFEbaseD2D_runtime/by_date/2026-07-19/comparisons/
  gbs32_50k_report_20260719
```

50k 下游任务只保留 LAMBADA 与 HellaSwag 作为主要趋势指标；ARC 和 WikiText 已
从后续 benchmark 中移除。

| 任务 | 指标 | Dense 50k | MoFE baseline 50k | Upcycling 50k | MoFE baseline - Dense | Upcycling - Dense |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LAMBADA OpenAI | acc | 0.342519 | 0.342325 | 0.340190 | -0.000194 | -0.002329 |
| HellaSwag | acc_norm | 0.315873 | 0.318861 | **0.319757** | +0.002987 | +0.003884 |

对 5k、10k、...、50k 的十个 checkpoint 逐点评测后，HellaSwag 上 MoFE
baseline 从中后段开始基本持续高于 Dense；LAMBADA 更抖，双方交替领先。完整
趋势表和图位于：

```text
/data/chenpengxu/MoFEbaseD2D_runtime/comparisons/
  gbs32_lambada_hellaswag_trend_5k_points_20260719
```
三方法 50k 对照的原始运行目录已按日期归档。当前 active runtime 只保留正在运行
的 MoFE group-LR 实验；历史 50k checkpoint 位于：

```text
/data/chenpengxu/MoFEbaseD2D_runtime/by_date/2026-07-17/runs/
  dense_gpt2_finewebedu10bt_4gpu_5000step_fp32master_constlr_nowarmup_val100_ckpt1000_20260717
  gbs32_dense5k10k_then_mofe10k_fp32master_constlr_nowarmup_val100_ckpt1000_20260717
/data/chenpengxu/MoFEbaseD2D_runtime/by_date/2026-07-18/runs/
  gbs32_dense10k50k_then_mofe10k50k_fp32master_constlr_nowarmup_val200_ckpt5000_20260718
/data/chenpengxu/MoFEbaseD2D_runtime/by_date/2026-07-19/runs/
  upcycling_gpt2_finewebedu10bt_4gpu_50k_gbs32_fp32master_constlr_nowarmup_val200_ckpt5000_20260719
```

### 2026-07-20 to 2026-07-21: MoFE group-LR finalization

原始 MoFE 使用统一 `1e-5` 学习率。最终版本改为参数组学习率：backbone 和 shared
expert 为 `1e-5`，private factorized experts 为 `2e-5`，router 为 `3e-5`；其他
训练协议与 Dense、Upcycling 保持一致。该版本从此成为项目中默认的 MoFE。

50k FineWeb-Edu held-out 结果：

| 模型 | Validation loss | PPL | Next-token accuracy |
| --- | ---: | ---: | ---: |
| Dense | 3.112968 | 22.4877 | 40.8511% |
| MoFE group LR | **3.092061** | **22.0224** | **41.1613%** |
| Upcycling | 3.104230 | 22.2921 | 40.9910% |

对 5k 到 50k 的每 5k checkpoint 测量 next-token prediction accuracy，共得到
30 个原始评测点。50k zero-shot 下游任务统一使用原始 `acc`：

| 模型 | LAMBADA acc | HellaSwag acc |
| --- | ---: | ---: |
| Dense | 0.340772 | 0.291675 |
| MoFE group LR | **0.343101** | **0.294662** |
| Upcycling | 0.340190 | 0.293467 |

最终 50k 原始数据归档在 `results/final_50k/`。ARC、WikiText 和原始 MoFE baseline
不再用于后续主实验或最终结果表。2026-07-21 启动三模型 50k 到 80k 的串行扩展，
保持 GBS 32、每 200 step validation、每 5k step 保存完整训练状态；该扩展结果
尚未并入冻结的 50k 数据归档。

### 2026-07-21: repository finalization

- 将原 README 中的逐日记录迁移到 `timeline.md`。
- 重写 README，使其只描述最终方法、实验协议、50k 结果和复现入口。
- 保留所有历史目录、历史结果和上游工程文件；最终 50k 数据另行集中归档。
- 归档 Dense、MoFE group LR、Upcycling 的 50k validation loss、30 个 validation
  prediction accuracy 原始点，以及两个下游任务的原始 lm-eval JSON。
