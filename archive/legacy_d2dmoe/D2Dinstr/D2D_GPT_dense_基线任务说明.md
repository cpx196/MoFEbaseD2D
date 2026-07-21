# D2DMoE GPT Dense 基线复现任务说明

## 1. 本阶段目标

搭建 D2DMoE 代码环境，加载 **dense GPT-2 small** 预训练模型，完成因果语言模型推理验证和标准下游测评，形成后续 MoE 实验的可复现基线。

本阶段**不进行** MoE 转换、router 训练或继续预训练。

后续将以该基线比较三类方法：

1. Dense 模型继续训练；
2. 常规 dense-to-MoE upcycling；
3. 我们的方法：由 A/B/core 生成新 expert，并保留原始 dense FFN 为 shared expert。

## 2. 固定实验配置

| 项目 | 固定要求 |
| --- | --- |
| 代码仓库 | `https://github.com/bartwojcik/D2DMoE.git` |
| 代码目录 | 放在服务器上便于管理和后续修改的位置；建议位于实验工作目录附近，但不强制固定路径 |
| 起始模型 | `openai-community/gpt2`，即 GPT-2 small |
| 模型规模 | 124M 参数，12 层，hidden size 为 768 |
| FFN 类型 | 两层 GELU MLP，不是 SwiGLU |
| 模型结构 | 全部保持 dense，不做 MoE 改造 |
| 精度 | GPU 支持时使用 bf16；冒烟测试允许 fp32 |
| 数据、缓存、日志、结果目录 | 当前路径下的 `./data/pxchen` |
| GPU 使用 | 先用单卡完成正确性和评测；多卡仅用于加速，可选 |

不要使用 `gpt2-medium`、`gpt2-large` 或 `gpt2-xl`。Hugging Face 中的 `gpt2` 指的就是 GPT-2 small。

## 3. 目录规范

所有模型缓存、数据集、日志和评测结果必须写入 `./data/pxchen`，不要写入默认的 `~/.cache`。

```text
./
└── data/pxchen/
    ├── hf/                            # Hugging Face 模型和 tokenizer 缓存
    ├── datasets/                      # 下游评测数据集缓存
    ├── runs/dense_gpt2_small/         # 运行日志、环境信息
    └── results/dense_gpt2_small/      # 机器可读的评测结果和图表
```

在当前项目根目录执行以下环境变量设置。之后凡是下载模型或数据集的 shell 都应设置这些变量：

```bash
export PROJECT_ROOT="$(pwd)"
export DATA_ROOT="$PROJECT_ROOT/data/pxchen"
export HF_HOME="$DATA_ROOT/hf"
export HF_DATASETS_CACHE="$DATA_ROOT/datasets"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" \
  "$DATA_ROOT/runs/dense_gpt2_small" \
  "$DATA_ROOT/results/dense_gpt2_small"
```

## 4. 代码与环境准备

在与服务器 CUDA 驱动匹配的干净 Python 环境中执行。代码仓库可以克隆在当前工作目录下，也可以放在服务器上其他便于管理的代码目录；不要放进 `./data/pxchen`。无论实际放在哪里，都需要将代码仓库的绝对路径、Python、PyTorch、CUDA、GPU、依赖版本和 D2DMoE commit 记录到 `environment.txt`。

```bash
# D2D_DIR 可按服务器目录结构自行设置，例如 "$PROJECT_ROOT/D2D"。
# 代码目录不固定，但必须在最终日志中记录 D2D_DIR 的绝对路径。
export D2D_DIR="${D2D_DIR:-$PROJECT_ROOT/D2D}"
git clone https://github.com/bartwojcik/D2DMoE.git "$D2D_DIR"
cd "$D2D_DIR"
git rev-parse HEAD | tee "$DATA_ROOT/runs/dense_gpt2_small/d2d_commit.txt"

# 先安装仓库依赖。仅在当前 PyTorch/CUDA 不可用时，才调整 PyTorch CUDA wheel。
pip install -r requirements.txt
pip install "transformers>=4.40" "datasets>=2.18" "safetensors" "lm-eval>=0.4,<0.5"

{
  date
  python --version
  python -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.version.cuda); print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'
  pip freeze
  printf 'D2D_DIR: %s\n' "$D2D_DIR"
  git rev-parse HEAD
} > "$DATA_ROOT/runs/dense_gpt2_small/environment.txt"
```

如果仓库没有 `requirements.txt`，或其依赖与已经可用的 CUDA PyTorch 环境冲突，不要覆盖可工作的 GPU PyTorch。安装其余依赖即可，并把最终解决方法记录在 `environment.txt`。

## 5. 加载 checkpoint 并完成推理冒烟测试

Hugging Face 页面显示约 6GB 是因为仓库中同时包含 TensorFlow、Flax、ONNX 以及重复序列化格式。我们只需 Transformers 所需的模型、配置和 tokenizer 文件；`model.safetensors` 的 fp32 权重约为 548MB。

在实际的 D2DMoE 代码目录，即 `$D2D_DIR`，执行：

```bash
python - <<'PY' | tee "$DATA_ROOT/runs/dense_gpt2_small/smoke_test.txt"
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai-community/gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
model = AutoModelForCausalLM.from_pretrained(
    model_id, cache_dir=os.environ["HF_HOME"], torch_dtype=dtype
).to(device).eval()

prompt = "The dense GPT-2 baseline is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print("model:", model_id)
print("device:", device)
print("dtype:", dtype)
print("parameter_count:", sum(p.numel() for p in model.parameters()))
print(tokenizer.decode(output[0], skip_special_tokens=True))
PY
```

验收要求：

1. 冒烟测试在 GPU 上正常结束，没有 OOM。
2. `parameter_count` 应约为 `124,439,808`。
3. 日志中有非空的确定性文本续写结果。
4. 模型和 tokenizer 缓存位于 `./data/pxchen/hf`。

## 6. 下游任务测评

使用 `lm-evaluation-harness` 做标准的零样本语言模型评测。本阶段得到的是原始 dense GPT-2 的基线结果，不能将其表述为 D2DMoE 的复现结果。

必须评测的任务如下：

| 任务 | 作用 |
| --- | --- |
| `lambada_openai` | 长上下文的词预测 / 语言模型补全 |
| `hellaswag` | 常识续写排序 |
| `piqa` | 物理常识推理 |
| `winogrande` | 指代消解和常识推理 |

全部使用零样本设置，即 `num_fewshot=0`：

```bash
lm_eval \
  --model hf \
  --model_args pretrained=openai-community/gpt2,dtype=bfloat16,trust_remote_code=False \
  --tasks lambada_openai,hellaswag,piqa,winogrande \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path "$DATA_ROOT/results/dense_gpt2_small/lm_eval" \
  2>&1 | tee "$DATA_ROOT/runs/dense_gpt2_small/lm_eval.log"
```

若当前版本的 `lm-eval` 不支持 `dtype=bfloat16` 或 `batch_size=auto`，可改用 `dtype=float32` 和显式 batch size，例如 `batch_size=4`，并在日志中记录最终实际命令。无论如何，评测对象必须是 dense GPT-2 small。

## 7. WikiText-2 Perplexity

除下游任务外，还需报告 WikiText-2 raw validation 的 perplexity。使用最大长度 1024、stride 512 的滑动窗口评测实现，结果保存为：

```text
./data/pxchen/results/dense_gpt2_small/wikitext2_validation.json
```

结果 JSON 至少包括：

```json
{
  "model": "openai-community/gpt2",
  "dataset": "wikitext/wikitext-2-raw-v1",
  "split": "validation",
  "max_length": 1024,
  "stride": 512,
  "precision": "bfloat16",
  "negative_log_likelihood": 0.0,
  "perplexity": 0.0
}
```

必须使用 causal LM 的正确 shifted labels 和按 token 数加权的 NLL。不要把不同长度文档 padding 成一个 batch 后直接平均 batch loss，否则 perplexity 会不正确。把实际使用的评测脚本位置和命令写入运行日志。

## 8. 汇总表和作图

除原始 JSON 外，需要整理一张便于后续 MoE 实验追加的汇总表，保存为：

```text
./data/pxchen/results/dense_gpt2_small/metrics_summary.csv
```

表格至少包含 `experiment`、`model`、`moe_layers`、`num_experts`、`top_k`、`task`、`metric`、`score`、`precision`、`d2d_commit`。本阶段的固定取值为 `experiment=dense_baseline`、`moe_layers=0`、`num_experts=0`、`top_k=0`。

需要生成两张 PNG 图，图中标题、坐标轴、图例均使用英文，便于后续论文或汇报直接使用：

1. `downstream_scores.png`：六个下游任务的 zero-shot 分数柱状图，横轴为任务名，纵轴为对应 accuracy / normalized accuracy；图标题标注 `GPT-2 small dense baseline`。
2. `baseline_overview.png`：一张简洁的基线总览图，左侧展示 WikiText-2 validation perplexity，右侧展示六项下游得分；必须在图中注明模型为 GPT-2 small、124M、dense、zero-shot。

图表保存位置：

```text
./data/pxchen/results/dense_gpt2_small/figures/
├── downstream_scores.png
└── baseline_overview.png
```

作图脚本也需保留在代码目录或实验脚本目录中，并在日志中注明其绝对路径和运行命令。数值必须直接读取 lm-eval JSON 和 WikiText-2 JSON，不允许手工转录，避免后续对比时出现不一致。

## 9. 最终交付物

完成后应有以下文件：

```text
./data/pxchen/runs/dense_gpt2_small/
├── d2d_commit.txt
├── environment.txt
├── smoke_test.txt
└── lm_eval.log

./data/pxchen/results/dense_gpt2_small/
├── lm_eval/                          # lm-evaluation-harness 输出的 JSON
├── wikitext2_validation.json
├── metrics_summary.csv
└── figures/
    ├── downstream_scores.png
    └── baseline_overview.png
```

同时给出一段简短的汇总，至少包含：使用的 checkpoint 标识符、D2DMoE commit hash、GPU 型号和数量、实际精度、各个下游任务分数、WikiText-2 validation perplexity。

## 10. 本阶段边界

本阶段不得修改模型为 MoE。尤其不能进行 FFN 拆分、dense FFN 复制为 experts、添加 shared expert、生成 A/B/core experts、训练 router，或提前报告 MoE 相对 dense 的结论。

后续所有 MoE 实验应沿用本阶段的模型、数据和评测协议，才能形成公平对比。

## 11. 参考链接

- D2DMoE 论文：<https://arxiv.org/abs/2405.15719>
- D2DMoE 代码：<https://github.com/bartwojcik/D2DMoE>
- Dense 起始 checkpoint：<https://huggingface.co/openai-community/gpt2>
- Sparse Upcycling 论文（后续基线参考）：<https://arxiv.org/abs/2212.05055>
