# 120k Diagnostic Experiment Archive

This directory contains the compact data archive for the Dense, MoFE group-LR,
and Upcycling runs through optimizer step 120,000. At 32,768 tokens per step,
the step counter corresponds to 3.93216B processed tokens per model.

## Important Data-Stream Limitation

These runs used a streaming FineWeb-Edu dataset. Model, optimizer, scheduler,
and RNG state were restored at continuation boundaries, but the dataloader
cursor was not saved. Each newly launched continuation therefore rebuilt the
fixed-seed shuffled stream from its beginning. In particular, the 50k to 80k
and 80k to 120k segments overlap in training examples; MoFE also restarted its
stream at 95k.

Consequently, 3.93216B is the number of processed tokens, not the number of
unique tokens. Treat this archive as a diagnostic continuation study, not as an
uninterrupted unique-token scaling curve. The held-out validation and downstream
evaluations themselves are valid for the saved checkpoints.

## Files

- `validation/validation_loss_120k.csv`: all fixed held-out loss records through
  step 120k for the three models.
- `validation_prediction_accuracy/validation_token_accuracy_5k_to_120k.csv`:
  72 evaluations, one every 5k steps for each model.
- `validation_prediction_accuracy/raw/*.json`: original token-accuracy outputs,
  including correct-token and evaluated-token counts.
- `downstream/downstream_80k_to_120k.csv`: consolidated LAMBADA and HellaSwag
  results at 80k, 100k, 110k, and 120k, including standard errors.
- `downstream/raw/*.json`: the 12 original lm-evaluation-harness outputs.
- `figures/validation_loss_and_token_accuracy_120k.png`: combined held-out loss
  and next-token accuracy plot.

## Step 120k Snapshot

| Model | Validation loss | Token accuracy | LAMBADA acc | HellaSwag acc |
| --- | ---: | ---: | ---: | ---: |
| Dense | 3.088866 | 0.411081 | 0.336115 | 0.292272 |
| MoFE group LR | **3.059887** | **0.414678** | 0.339220 | **0.295758** |
| Upcycling | 3.072386 | 0.413044 | **0.339802** | 0.294563 |

Token accuracy is evaluated on 242,451 held-out predicted tokens. Downstream
evaluation uses lm-evaluation-harness 0.4.12, zero-shot, batch size 4, bfloat16,
fixed seeds, and 100 bootstrap iterations. Dataset sizes are 5,153 examples for
LAMBADA OpenAI and 10,042 examples for HellaSwag. The primary HellaSwag metric
in this project is unnormalized `acc`; `acc_norm` is retained in the CSV and raw
JSON files.
