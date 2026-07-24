# Final 50k Experiment Data

This directory contains the compact raw-data archive for the final matched
Dense, MoFE group-LR, and Upcycling experiments. Every model was trained for
50,000 optimizer steps, or 1.6384B training tokens, on FineWeb-Edu 10BT.

## Data-Stream Limitation

The streaming dataloader cursor was not stored in continuation checkpoints.
When training resumed in a new process, the fixed-seed shuffled stream was
rebuilt from its beginning. The reported token budget therefore counts
processed tokens and may include repeated examples across continuation
boundaries; it is not a guaranteed unique-token count. The checkpoint
evaluations remain valid, but this archive should not be interpreted as an
uninterrupted unique-token scaling run.

## Files

- `validation/validation_loss_50k.csv`: all available held-out validation-loss
  records through step 50k. Columns are model, optimizer step, training tokens
  in billions, validation loss, and perplexity. The obsolete MoFE baseline has
  been removed.
- `validation_prediction_accuracy/validation_token_accuracy_5k_points.csv`:
  summary of 30 evaluations: Dense, MoFE group LR, and Upcycling at 5k, 10k,
  ..., 50k.
- `validation_prediction_accuracy/raw/*.json`: the 30 original evaluation
  outputs, including correct-token counts and total evaluated tokens.
- `downstream/*.json`: original lm-evaluation-harness outputs at step 50k for
  Dense, MoFE group LR, and Upcycling on `lambada_openai` and `hellaswag`.
- `figures/validation_loss_and_token_accuracy_50k.png`: the final side-by-side
  held-out loss and next-token accuracy figure.

Next-token accuracy is computed by comparing
`argmax(logits[:, :-1])` with `labels[:, 1:]`, excluding padding labels. The
fixed held-out set contains 242,451 predicted tokens.

The primary downstream metric is unnormalized `acc` for both tasks. Raw
HellaSwag files also contain `acc_norm` because lm-evaluation-harness reports it
by default.
