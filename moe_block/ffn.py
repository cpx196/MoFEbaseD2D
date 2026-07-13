import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class FFN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
        moe_num_expert=32,
        moe_top_k=2,
        # gate=NaiveGate,
        # gate=GShardGate,
    ):
        super().__init__()
        self.num_expert = moe_num_expert
        self.d_model = dim_feedforward

        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

    def forward(self, input):
        tgt = self.linear1(input)
        tgt = self.dropout(self.activation(tgt))

        output = self.linear2(tgt)

        return output