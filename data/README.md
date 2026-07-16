# External Data And Checkpoints

Large experiment artifacts are intentionally stored outside this source tree.
On the current server they were moved, without deletion, to:

```text
/home/iot-mengshiyuan/MoFEbaseD2D_data/pxchen
```

That directory contains datasets, Hugging Face caches, local GPT-2 weights,
training checkpoints, optimizer states, complete logs, and evaluation results.

Recommended environment variables:

```bash
export DATA_ROOT=/home/iot-mengshiyuan/MoFEbaseD2D_data/pxchen
export HF_HOME="$DATA_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

The source archive does not include those files. Copy only the required data or
checkpoints separately when moving the project to another machine.
