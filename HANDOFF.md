# MoFEbaseD2D Session Handoff

Updated: 2026-07-16 UTC

This document contains the project context needed to continue the work in a new
assistant session. It intentionally excludes raw conversation records, Codex
internal state, credentials, and large experiment artifacts.

## Objective

The project compares three GPT-2 Small continued-pretraining methods on matched
WikiText-103 data and optimizer steps:

1. Dense GPT-2 baseline.
2. MoFE: one always-on shared FFN plus token-routed factorized private experts.
3. Sparse Upcycling: replace the last three FFNs with routed, complete copies.

The primary scientific questions are whether factorized private experts improve
sample efficiency, whether routing learns useful token specialization, and how
MoFE compares with a conventional dense-FFN Upcycling control.

## Repository And Artifacts

Source repository:

```text
/home/iot-mengshiyuan/MoFEbaseD2D
```

Large files were moved without deletion to:

```text
/home/iot-mengshiyuan/MoFEbaseD2D_data/pxchen
```

Important subdirectories:

```text
checkpoints/  # final weights and resumable optimizer states
datasets/     # WikiText-103 and datasets cache
hf/           # Hugging Face cache
hf_models/    # local GPT-2 model
results/      # complete generated reports and figures
runs/         # console and evaluation logs
```

Recommended environment:

```bash
export DATA_ROOT=/home/iot-mengshiyuan/MoFEbaseD2D_data/pxchen
export HF_HOME="$DATA_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The Git worktree contains intentional, uncommitted implementation and analysis
changes. Do not reset or discard them.

## MoFE Architecture

Only GPT-2 blocks 9, 10, and 11 are converted. Each converted MLP contains:

- One complete shared GPT-2 FFN, always active.
- 16 factorized private experts.
- Token-choice top-3 routing over private experts.
- Four A groups and four B groups combined as a 4x4 Cartesian product.
- Rank 576, equal to `0.75 * hidden_size` for GPT-2 hidden size 768.

For expert `e = 4i + j`:

```text
W1_e = A1_i C1_e B1_j
W2_e = A2_i C2_e B2_j
output = shared(x) + private_scale * routed_private(x)
```

Current initialization:

- A/B are copied from dense FFN row/column slices.
- `C1 ~ Normal(0, 0.025^2)` independently per expert.
- `C2 = 0`.
- Private biases are zero.
- Router weights use `Normal(0, 0.02^2)`.

Important provenance: `C2=0` was added in commit `c2059bb` as a stabilization
change after an exploding-loss run. The earlier implementation initialized both
cores with `Normal(0, 0.025^2)`. Zeroing only the output core is an engineering
choice, not an established requirement inherited from the original method.

There are two core tensors because GPT-2 FFN has two projections. Each tensor
contains 16 matrices, so the last three MoFE layers contain `3 * 16 * 2 = 96`
individual core matrices.

## Key Training Fixes

- Accelerate scheduler multiplication was disabled. The scheduler now advances
  once per real optimizer update, not once per process.
- Constant LR was added: 10-step warmup followed by `1e-5`.
- Training loss logging was corrected. Each logged value is the cross-rank,
  cross-gradient-accumulation average over effective batch 32, not rank 0's last
  microbatch.
- Fixed WikiText-103 validation runs every 100 optimizer steps on 240 sequences
  of 1024 tokens, totaling 245,520 predicted tokens.
- MoFE private output scale is linearly increased from 0 to 1 over 100 steps.
- Checkpoints saved through Accelerate contain model, optimizer, scheduler, and
  per-rank RNG state and can be resumed.

## Matched 1000-Step Experiment

Common configuration:

```text
WikiText-103 train
sequence length: 1024
2 GPUs
batch: 4/GPU
gradient accumulation: 4
effective batch: 32
tokens/optimizer step: 32,768
optimizer steps: 1000
LR: 10-step warmup, then constant 1e-5
weight decay: 0.1
seed: 42
```

Fixed WikiText-103 validation results:

| Model | Parameters | Loss | PPL |
| --- | ---: | ---: | ---: |
| Dense | 124,439,808 | 3.315577 | 27.5383 |
| MoFE E16 K3 | 209,595,696 | 3.113023 | 22.4889 |
| Upcycling E16 K3 | 336,986,160 | 3.316069 | 27.5518 |

Each training log has 201 effective-batch samples: step 1 and every 5 steps
through step 1000. Curated figures and CSV files are under:

```text
results/2026-07-15_1000step_comparison
```

## MoFE Private Ablation

The verified ablation uses the same step-1000 checkpoint and keeps the shared
branch enabled in both configurations.

| Configuration | WT103 loss | WT103 PPL | WT2 PPL |
| --- | ---: | ---: | ---: |
| Private OFF, scale 0 | 3.499410 | 33.0959 | 29.0748 |
| Private ON, scale 1 | 3.112870 | 22.4855 | 20.8666 |

Private ON reduces WT103 PPL by 32.06% and WT2 PPL by 28.23% relative to OFF.
This demonstrates predictive contribution from the private branch, but it is not
a percentage decomposition of model capability.

Important checkpoint:

```text
$DATA_ROOT/checkpoints/mofe_gpt2_wikitext103_zero_core2_1000step_b4_2gpu_fixedval
```

The project also contains MoFE 300, 1500, and 3000-step checkpoints and earlier
zero-shot evaluations under the external artifact directory.

## Sparse Upcycling Control

Implementation files:

```text
MoFE/upcycling.py
MoFE/train_upcycling.py
MoFE/configs/upcycling_gpt2_last3_e16_k3.json
MoFE/tests/test_upcycling.py
```

Blocks 9, 10, and 11 each replace the original FFN with 16 independent complete
copies. A token-choice router selects top-3 experts and normalizes their weights.
There is no additional shared expert.

The step-1000 checkpoint includes a resumable state:

```text
$DATA_ROOT/checkpoints/upcycling_gpt2_wikitext103_e16_k3_1000step_b4_2gpu_fixedval/step_001000
```

Upcycling did not outperform Dense at 1000 steps. Diagnostics found:

- Expert relative parameter divergence is only about 0.11%--0.15%.
- Router entropy stays close to `log(16)`, so routing remains weakly confident.
- Layer 9 has persistent load imbalance; its average max/min expert share ratio
  is about 14.18.
- All experts are used; forward, gradients, checkpoint round-trip, and step-0
  function preservation checks pass.
- All three historical matched experiments use BF16 model parameters and BF16
  Adam moments. A later FP32-master control showed that this creates an
  architecture-dependent bias: nonzero pretrained Dense weights frequently lose
  small updates to BF16 rounding, while zero-initialized MoFE cores can still
  update. Those historical relative results are not valid final comparisons.

The leading explanation is insufficient expert differentiation under a short,
homogeneous training budget, not a discovered forward-pass bug.

## Routing Interpretation

MoFE's better result does not prove that its router learned clean semantic token
specialization. Its router entropy also remains near maximum. MoFE private
experts break symmetry earlier because Core1 is independently random, while
Upcycling experts start as identical functions. MoFE additionally has an
always-on shared FFN and factor banks shared across private experts, which makes
short-budget training more sample-efficient.

The next necessary routing experiment is a router-shuffle ablation: preserve all
trained weights but randomly permute or replace top-3 assignments at evaluation.
A significant PPL degradation would isolate predictive value from learned token
routing rather than generic private capacity.

## FineWeb-Edu 10BT

The local continued-pretraining dataset is:

```text
repository: HuggingFaceFW/fineweb-edu
config path: sample/10BT
split: train
local path: /data/chenpengxu/HMoE_data/fineweb_10BT
```

It was verified on 2026-07-16:

```text
Parquet shards: 14
compressed bytes: 28,518,193,415 (26.56 GiB)
documents: 9,672,101
sum(token_count): 9,972,918,177
text column: text
```

All Parquet footers are readable and all shard schemas match. The downloader
validated each file against its API-reported byte size, but no pinned repository
revision, `manifest.json`, or `SHA256SUMS` exists yet.

`MoFE/train.py`, `MoFE/train_dense.py`, and `MoFE/train_upcycling.py` now accept a
directory of Parquet shards through `--train-file`. Directory inputs are sorted
by filename and automatically streamed. Streaming uses deterministic buffered
shuffle with `--seed`, followed by batched tokenization and 1024-token packing;
the existing JSON/JSONL WikiText path remains map-style.

A four-RTX-4090 MoFE smoke run completed successfully:

```text
output: /data/chenpengxu/MoFEbaseD2D_runtime/runs/mofe_gpt2_fineweb_4gpu_smoke_20260716
optimizer steps: 2
effective batch: 4
tokens per optimizer step: 4,096
step 1 LM loss: 3.5921
step 2 LM loss: 3.8541
peak allocated memory: 3.77 GiB/GPU
```

This proves the data and four-GPU training path, not convergence. The smoke run
used batch size 1 per GPU, no gradient accumulation, constant `5e-5` learning
rate, no private warmup, and a small shuffle buffer of 256 documents. Choose the
matched long-run budget and warmup separately.

### FineWeb-Edu 1000-Step Result

Precision warning: this entire 1000-step run used BF16 master parameters and
BF16 Adam moments. It is retained for provenance, but its Dense/MoFE gap is
precision-confounded and must not be used as the active architecture comparison.
The FP32-master 200-step control below supersedes it.

A matched-effective-batch MoFE run completed on four RTX 4090 GPUs:

```text
output: /data/chenpengxu/MoFEbaseD2D_runtime/runs/mofe_gpt2_finewebedu10bt_4gpu_1000step_b4_ga2_fixedval_sb2048_nw0_20260716
sequence length: 1024
batch: 4/GPU
gradient accumulation: 2
effective batch: 32
tokens/optimizer step: 32,768
optimizer steps: 1000
LR: 10-step warmup, then constant 1e-5
private scale warmup: 100 steps
shuffle buffer: 2,048 documents
training DataLoader workers: 0
final throughput: 43,658 tokens/s
peak allocated memory: 10.58 GiB/GPU
```

The first attempt used a 10,000-document buffer and two workers. Four streaming
workers each reached about 15 GiB RSS and the Linux OOM killer stopped the run at
step 35. Streaming training now forces `num_workers=0`, and the default buffer is
2,048. The successful run used about 27 GiB total host RAM and saved resumable
states every 100 steps. The output directory is about 13 GiB including all ten
states and the final evaluation checkpoint.

The active evaluation target is the held-out FineWeb-Edu split:

| Metric | GPT-2 start | Final private OFF | Final private ON |
| --- | ---: | ---: | ---: |
| FineWeb-Edu held-out loss | 3.299973 | 3.300422 | **3.235094** |
| FineWeb-Edu held-out PPL | 27.1119 | 27.1241 | **25.4088** |

On held-out FineWeb-Edu, private ON lowers loss by 0.064879 and PPL by 6.28%
relative to the original GPT-2. Private OFF is essentially unchanged, so the
private branch carries the in-domain gain.

All ten saved checkpoints were retrospectively evaluated on the same held-out
set. Validation loss decreases at every checkpoint from step 100 (`3.264837`)
through step 900 (`3.234236`), then increases slightly by `0.000859` at step
1000 (`3.235094`). Step 900 is the best saved checkpoint on this validation set.
The curve and exact table are under `reports/fineweb_validation_curve` in the
1000-step run directory.

A matched four-GPU Dense GPT-2 run was completed on 2026-07-17 with the same
FineWeb-Edu stream, seed, sequence length, effective batch 32, constant `1e-5`
learning rate, and 1000 optimizer steps. Under a common checkpoint replay
protocol, Dense validation loss changes from `3.299059` to `3.280629` (PPL
`27.0871` to `26.5925`, -1.83%), while MoFE reaches loss `3.235094` and PPL
`25.4088`. MoFE final PPL is 4.45% lower than Dense. Dense reaches 83,297
tokens/s and 7.46 GiB/GPU peak allocation versus MoFE's 43,658 tokens/s and
10.58 GiB/GPU. Dense output and comparison reports are under:

```text
/data/chenpengxu/MoFEbaseD2D_runtime/runs/
  dense_gpt2_finewebedu10bt_4gpu_1000step_b4_ga2_finewebval_sb2048_nw0_20260717
/data/chenpengxu/MoFEbaseD2D_runtime/comparisons/
  dense_mofe_finewebedu_4gpu_1000step_20260717
```

The Dense and MoFE step-900 checkpoints were evaluated zero-shot with lm-eval
0.4.12 on LAMBADA OpenAI, HellaSwag, PIQA, and WinoGrande. MoFE wins two tasks
and Dense wins two. Their four-task macro averages are `0.438096` and
`0.438073`, respectively, a MoFE difference of only `+0.000023`. Every accuracy
difference is smaller than the reported standard
errors, so this run does not establish a downstream winner. MoFE lowers LAMBADA
perplexity by 0.86% while its exact-match accuracy is slightly lower. Full raw
results and summaries are in the comparison directory's `downstream` directory.

CalFLOPs 0.3.2 profiling at sequence length 1024 reports `291.898 GFLOPs` per
Dense forward and `385.741 GFLOPs` per MoFE forward for one sequence, a
`1.3215x` ratio. Profiling forces eager attention because CalFLOPs cannot see
the QK^T and AV matmuls inside fused SDPA. With 32,768 tokens per optimizer step
and backward estimated as twice the forward cost, training costs are
`28.022 TFLOPs/step` for Dense and `37.031 TFLOPs/step` for MoFE. Thus 1000
MoFE steps are compute-matched by about `1321.5` Dense steps. The reproducible
entry point is `python -m MoFE.profile_flops`; full output is under the
comparison directory at `compute/calflops_profile.json`.

A corrected matched 200-step no-warmup control was then run on four GPUs. Both models use
constant LR `1e-5` from the first optimizer step, effective batch 32, sequence
length 1024, and validation every 10 steps. MoFE also uses private scale 1 from
step 0 with no private warmup. Training is restricted to FineWeb-Edu shards
000--012, while validation is the fixed shard-013 held-out file. Each model sees
6,553,600 training tokens. Master parameters and AdamW moments remain FP32 while
forward/backward compute uses BF16 autocast. Dense validation loss changes from
`3.299973` to `3.221984` (PPL `25.0778`); MoFE changes from `3.299973` to
`3.211319` (PPL `24.8118`). At step 200 MoFE loss is lower by `0.010665` and PPL
is lower by 1.06%. Both best validation points are step 200. The prior BF16-master
200-step control reported a 4.38% gap but is invalidated by this precision fix.
Full 10-step logs, plots, CSV, and summary are under:

```text
/data/chenpengxu/MoFEbaseD2D_runtime/comparisons/
  dense_mofe_finewebedu_4gpu_200step_fp32master_constlr_nowarmup_val10_20260717
```

The corrected step-200 checkpoints were evaluated zero-shot with lm-eval 0.4.12
on LAMBADA OpenAI, HellaSwag, PIQA, and WinoGrande. Dense scores are `0.320396`,
`0.313483`, `0.630033`, and `0.513812`; MoFE scores are `0.319814`, `0.312189`,
`0.621328`, and `0.501973`. Four-task macro averages are `0.444431` for Dense and
`0.438826` for MoFE. Dense is numerically higher on all four tasks, but every
individual difference is smaller than the reported independent standard error,
so this run does not establish a statistically clear downstream winner. MoFE's
LAMBADA perplexity is 1.82% higher. Raw JSON, CSV, figure, and report are in the
comparison directory's `downstream` subdirectory.

The missing Sparse Upcycling E16 K3 control was rerun under the same corrected
200-step protocol: FP32 master parameters and AdamW states, BF16 compute,
constant LR `1e-5`, no warmup, effective batch 32, 6,553,600 training tokens,
and validation every 10 steps. Its 336,986,160 parameters comprise 16 complete
copied FFNs in each of blocks 9--11 with token-choice top-3 routing. Final loss
and PPL are `3.225175` and `25.1580`, compared with Dense `3.221984`/`25.0778`
and MoFE `3.211319`/`24.8118`. Upcycling is `+0.003191` loss above Dense and
`+0.013856` above MoFE, while using 14.88 GiB/GPU and reaching 29,053 tokens/s.
All experts are used, but router entropy remains `2.7665--2.7670`, close to
`log(16)=2.7726`, and mean expert divergence relative to expert 0 is only
0.17%--0.21%. Full three-model artifacts are under:

```text
/data/chenpengxu/MoFEbaseD2D_runtime/comparisons/
  dense_mofe_upcycling_finewebedu_4gpu_200step_fp32master_constlr_nowarmup_val10_20260717
```

Evaluation policy as of 2026-07-17: do not use WikiText loss or perplexity for
training validation, checkpoint selection, or method conclusions. Existing
WikiText artifacts are historical only and should not be extended. Future fixed
validation data must be a pinned, disjoint FineWeb-Edu held-out split.

## Verification

Run from the repository with the `mofe` Conda environment:

```bash
/data/chenpengxu/conda_envs/HMoE/bin/python -m unittest \
  MoFE.tests.test_data MoFE.tests.test_mofe MoFE.tests.test_upcycling
```

There are currently 13 passing unit tests. Also run:

```bash
git diff --check
```

## Recommended Next Steps

1. Generate a pinned FineWeb-Edu manifest and checksums for provenance.
2. Define matched four-GPU Dense/MoFE/Upcycling budgets and run a longer pilot.
3. Implement and run MoFE router-shuffle evaluation on the fixed FineWeb-Edu
   held-out set.
4. Decide whether the experiment should retain `C2=0` stabilization or reproduce
   the earlier two-random-core initialization as a separate controlled run.
5. Run longer, matched Dense/MoFE/Upcycling experiments and multiple seeds before
   making final method claims.
