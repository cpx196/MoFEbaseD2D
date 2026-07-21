# Verified Step-1000 Private-branch Ablation

- Checkpoint global step: `1000`
- Checkpoint: `/home/iot-mengshiyuan/MoFEbaseD2D/data/pxchen/checkpoints/mofe_gpt2_wikitext103_zero_core2_1000step_b4_2gpu_fixedval/final`
- model_state.pt SHA256: `7766a0e79437652265a34c1fd1f3133db1fe5b75d89cb535c18293c07cf076d8`
- Shared scale remains `1` in both runs.

| Configuration | Private scale | WT103 loss | WT103 PPL | WT2 PPL |
| --- | ---: | ---: | ---: | ---: |
| Private OFF | 0 | 3.499410 | 33.0959 | 29.0748 |
| Private ON | 1 | 3.112870 | 22.4855 | 20.8666 |

Private ON reduces WT103 PPL by `32.06%` and WT2 PPL by `28.23%` relative to Private OFF.
