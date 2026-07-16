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
- All three matched experiments use BF16 model parameters and BF16 Adam moments.
  This is a shared controlled setting and does not explain Upcycling's relative
  result by itself.

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

## FineWeb Plan

The next proposed dataset is the Parquet-sharded configuration:

```text
repository: HuggingFaceFW/fineweb
config: sample-10BT
split: train
```

FineWeb 10BT is expected to be about 25--35 GB as original Parquet and should be
downloaded outside the source repository. The new upload target is:

```text
$DATA_ROOT/datasets/fineweb/sample-10BT
```

The local downloader should preserve original Parquet shards, pin the repository
commit, and produce `manifest.json` plus `SHA256SUMS`. Do not convert the full
dataset into one JSONL file. After upload, verify checksums before changing the
training loader to accept multiple Parquet shards.

## Verification

Run from the repository with the `mofe` Conda environment:

```bash
/home/iot-mengshiyuan/conda_envs/mofe/bin/python \
  -m unittest MoFE.tests.test_mofe MoFE.tests.test_upcycling
```

There are currently 11 passing unit tests. Also run:

```bash
git diff --check
```

## Recommended Next Steps

1. Verify and register the uploaded FineWeb Parquet manifest and checksums.
2. Add deterministic multi-Parquet loading without changing existing WikiText
   preprocessing behavior.
3. Implement and run MoFE router-shuffle evaluation on the fixed validation set.
4. Decide whether the experiment should retain `C2=0` stabilization or reproduce
   the earlier two-random-core initialization as a separate controlled run.
5. Run longer, matched Dense/MoFE/Upcycling experiments and multiple seeds before
   making final method claims.
