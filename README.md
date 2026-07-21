# MoFEbaseD2D

MoFEbaseD2D studies parameter-efficient sparse expansion of GPT-2 Small on
FineWeb-Edu 10BT. It compares three models under the same training-token budget:

- **Dense**: continued pretraining of GPT-2 Small.
- **MoFE**: Mixture of Factorized Experts in the final three GPT-2 MLP blocks.
- **Upcycling**: sparse MoE expansion using complete copies of the final three MLPs.

In this repository, **MoFE means the group-learning-rate version** unless a
historical result is explicitly labeled `MoFE baseline`. The final MoFE setup uses
learning rates `1e-5` for the backbone/shared experts, `2e-5` for private
factorized experts, and `3e-5` for routers. The original single-learning-rate
MoFE is retained only in [timeline.md](timeline.md) as experiment history and is
not part of the final comparison.

This work is an independent derivative of
[D2DMoE](https://github.com/bartwojcik/D2DMoE), based on upstream commit
`a7027cdc1f01c9c618c39eebe639d1664549b066`. The upstream project and this
derivative use the MIT License.

## Method

MoFE replaces MLPs in Transformer blocks 9, 10, and 11. Each replacement keeps
the original dense MLP as an always-active shared expert and adds 16 private
factorized experts. A token-choice router selects the top 3 private experts per
token. The private experts share four input and four output factor banks through
a Cartesian `4 x 4` construction while retaining expert-specific cores.

For expert `e = 4i + j`, hidden size `d = 768`, MLP size `f = 3072`, and rank
`r = 576`, the two private projections are:

```text
W1_e = A1_i C1_e B1_j
W2_e = A2_i C2_e B2_j
```

Execution applies each factorized projection as `x -> B -> C -> A`; complete
expert matrices are never materialized per token. The converted layer output is:

```text
y = shared_mlp(x) + sum(g_e(x) * private_e(x))
```

Selected top-3 router weights are normalized to sum to one. There is no expert
capacity limit and no token dropping. Training adds balance and router-z terms:

```text
L = L_LM + 0.01 * L_balance + 0.001 * L_z
```

The shared expert copies the original GPT-2 MLP. Input-side private cores are
initialized from `Normal(0, 0.025^2)`; output-side cores and private biases start
at zero. Consequently, private outputs are exactly zero at initialization and
the converted model initially preserves the Dense GPT-2 function. Router weights
use `Normal(0, 0.02^2)` and zero bias.

All three models start from `openai-community/gpt2` and keep FP32 master
parameters and FP32 AdamW states. Forward and backward computation use BF16.

| Model | Parameters | Relative to Dense |
| --- | ---: | ---: |
| Dense | 0.124440B | 1.00x |
| MoFE | 0.209596B | 1.68x |
| Upcycling | 0.336986B | 2.71x |

## Main Experiment

The final archived comparison uses FineWeb-Edu 10BT with training shards
`000-012` and a fixed, disjoint held-out tail from shard `013`.

| Setting | Value |
| --- | --- |
| Hardware | 4 x RTX 4090 |
| Sequence length | 1024 |
| Per-device batch | 4 |
| Gradient accumulation | 2 |
| Global batch size | 32 sequences |
| Tokens per optimizer step | 32,768 |
| Optimizer steps | 50,000 |
| Training tokens per model | 1.6384B |
| Scheduler | Constant, no warmup |
| Weight decay | 0.1 |
| Validation interval | 200 steps |
| Checkpoint interval | 5,000 steps |
| Seed | 42 |

Checkpoints include model weights, optimizer and scheduler states, tokenizer,
and one RNG state per rank, so training can be resumed without rebuilding the
optimizer.

## Results at 50k

![Validation loss and next-token accuracy](results/final_50k/figures/validation_loss_and_token_accuracy_50k.png)

FineWeb-Edu held-out results:

| Model | Validation loss | Perplexity | Next-token accuracy |
| --- | ---: | ---: | ---: |
| Dense | 3.112968 | 22.4877 | 40.8511% |
| MoFE | **3.092061** | **22.0224** | **41.1613%** |
| Upcycling | 3.104230 | 22.2921 | 40.9910% |

MoFE has the lowest validation loss and highest next-token accuracy. Relative to
Dense, its final validation loss is lower by `0.020907`, while next-token
accuracy is higher by `0.3102` percentage points.

Zero-shot downstream results use the original `acc` metric for both tasks:

| Model | LAMBADA acc | HellaSwag acc |
| --- | ---: | ---: |
| Dense | 0.340772 | 0.291675 |
| MoFE | **0.343101** | **0.294662** |
| Upcycling | 0.340190 | 0.293467 |

ARC and WikiText are not part of the final benchmark. HellaSwag `acc_norm` is
present in the raw lm-eval output but is not used in the primary table.

## Archived Results

The compact, submission-ready raw data is stored in
[`results/final_50k/`](results/final_50k/):

```text
results/final_50k/
├── validation/                         # Complete held-out loss series
├── validation_prediction_accuracy/     # 30 raw JSON points and summary CSV
├── downstream/                         # Three raw 50k lm-eval JSON files
└── figures/                            # Final combined figure
```

See [results/final_50k/README.md](results/final_50k/README.md) for schemas and
file-level details. Checkpoints, datasets, caches, and full logs remain outside
Git under `/data/chenpengxu/MoFEbaseD2D_runtime/`.

## Code Layout

| Path | Purpose |
| --- | --- |
| `MoFE/layer.py` | Factorized experts, shared expert, and top-k routing |
| `MoFE/modeling.py` | GPT-2 conversion and parameter accounting |
| `MoFE/train.py` | MoFE training, including grouped learning rates |
| `MoFE/train_dense.py` | Matched Dense training |
| `MoFE/upcycling.py` | Sparse Upcycling model conversion |
| `MoFE/train_upcycling.py` | Matched Upcycling training |
| `MoFE/eval_validation_loss.py` | Fixed held-out loss evaluation |
| `MoFE/eval_validation_token_accuracy.py` | Next-token accuracy evaluation |
| `MoFE/eval_any_checkpoint.py` | LAMBADA and HellaSwag evaluation |
| `MoFE/configs/` | Final E16/K3 model configurations |
| `MoFE/tests/` | Data, MoFE, and Upcycling tests |

## Usage

Create an environment, install the single dependency list, then run the offline
tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest \
  MoFE.tests.test_data \
  MoFE.tests.test_mofe \
  MoFE.tests.test_upcycling
```

The final MoFE learning-rate arguments are:

```bash
python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m MoFE.train \
  --mofe-config MoFE/configs/mofe_gpt2_last3_e16_k3.json \
  --learning-rate 1e-5 \
  --shared-learning-rate 1e-5 \
  --private-learning-rate 2e-5 \
  --router-learning-rate 3e-5 \
  --per-device-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --validation-steps 200 \
  --save-steps 5000 \
  <data and output arguments>
```

The complete development and experiment record is maintained in
[timeline.md](timeline.md).

## Citation

The upstream D2DMoE project accompanies:

> Filip Szatkowski, Bartosz Wojcik, Mikolaj Piorczynski, Simone Scardapane.
> Exploiting Activation Sparsity with Dense to Dynamic-k Mixture-of-Experts
> Conversion. NeurIPS 2024. <https://arxiv.org/abs/2310.04361>
