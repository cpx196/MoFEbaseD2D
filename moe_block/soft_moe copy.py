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

class _Expert(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
    ):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

    def forward(self, x):
        tgt = self.linear1(x)
        tgt = self.dropout(self.activation(tgt))

        output = self.linear2(tgt)

        return output

class Soft_MoE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
        moe_num_expert,
        moe_top_k,
        expert_choice,
    ):
        super().__init__()
        self.num_expert = moe_num_expert
        self.d_model = dim_feedforward
        self.k = moe_top_k
        self.experts = nn.ModuleList([_Expert(hidden_dim, dim_feedforward, dropout, activation) for _ in range(moe_num_expert)])
        self.base_expert = _Expert(hidden_dim, dim_feedforward, dropout, activation)

        self.p = 8
        self.Phi = nn.Parameter(torch.zeros(hidden_dim, moe_num_expert, self.p)) # (d, n, p)
        nn.init.normal_(self.Phi, mean=0.0, std=1e-3)

        self.raw_g = nn.Parameter(torch.tensor(-10.0))

    def init_experts_from_checkpoint(self, state_dict, prefix: str):
        """
        state_dict: checkpoint['state_dict'] 或类似 dict
        prefix: 字符串前缀，例如
            'module.reconstruction.transformer.encoder.layers.0.new_moe_layer.'
        会在这个 prefix 下查找:
            prefix + 'linear1.weight' / 'linear1.bias'
            prefix + 'linear2.weight' / 'linear2.bias'
        并把找到的参数复制到 self.experts[*].linear1/linear2 对应的 weight/bias。
        """
        # keys we expect
        k_l1_w = prefix + 'mlp.fc1.weight'
        k_l1_b = prefix + 'mlp.fc1.bias'
        k_l2_w = prefix + 'mlp.fc2.weight'
        k_l2_b = prefix + 'mlp.fc2.bias'

        # try flexible matching if keys not found exactly
        def find_tensor(k):
            if k in state_dict:
                return state_dict[k]
            # try without 'module.' prefix
            if k.startswith('module.') and k[len('module.'): ] in state_dict:
                return state_dict[k[len('module.'):]]
            # try suffix-only keys
            suf = k.split('.')[-2] + '.' + k.split('.')[-1] if len(k.split('.'))>=2 else k
            if suf in state_dict:
                return state_dict[suf]
            return None

        t_l1_w = find_tensor(k_l1_w)
        t_l1_b = find_tensor(k_l1_b)
        t_l2_w = find_tensor(k_l2_w)
        t_l2_b = find_tensor(k_l2_b)

        if t_l1_w is None and t_l2_w is None:
            print(f"init_experts_from_checkpoint: no matching linear weights found for prefix '{prefix}'")
            return

        # copy linear1
        if t_l1_w is not None:
            if tuple(t_l1_w.shape) != tuple(self.base_expert.linear1.weight.shape):
                raise RuntimeError(f"shape mismatch for linear1.weight: ckpt {tuple(t_l1_w.shape)} vs expert {tuple(self.base_expert.linear1.weight.shape)}")
            self.base_expert.linear1.weight.data.copy_(t_l1_w.to(self.base_expert.linear1.weight.device))
        if t_l1_b is not None:
            if tuple(t_l1_b.shape) != tuple(self.base_expert.linear1.bias.shape):
                raise RuntimeError(f"shape mismatch for linear1.bias: ckpt {tuple(t_l1_b.shape)} vs expert {tuple(self.base_expert.linear1.bias.shape)}")
            self.base_expert.linear1.bias.data.copy_(t_l1_b.to(self.base_expert.linear1.bias.device))

        # copy linear2
        if t_l2_w is not None:
            if tuple(t_l2_w.shape) != tuple(self.base_expert.linear2.weight.shape):
                raise RuntimeError(f"shape mismatch for linear2.weight: ckpt {tuple(t_l2_w.shape)} vs expert {tuple(self.base_expert.linear2.weight.shape)}")
            self.base_expert.linear2.weight.data.copy_(t_l2_w.to(self.base_expert.linear2.weight.device))
        if t_l2_b is not None:
            if tuple(t_l2_b.shape) != tuple(self.base_expert.linear2.bias.shape):
                raise RuntimeError(f"shape mismatch for linear2.bias: ckpt {tuple(t_l2_b.shape)} vs expert {tuple(self.base_expert.linear2.bias.shape)}")
            self.base_expert.linear2.bias.data.copy_(t_l2_b.to(self.base_expert.linear2.bias.device))

        print(f"init_experts_from_checkpoint: copied params to the base experts from prefix '{prefix}'")

    def forward(self, x):
        a, b, c = x.shape
        x_flat = x.reshape(-1, c).contiguous()  # tokens x c

        n = self.num_expert
        p = self.p
        d = c
        m = x_flat.size(0) # tokens

        # logits: [m, n, p]
        logits = torch.einsum('md,dnp->mnp', x_flat, self.Phi) # [m, n, p]

        D = F.softmax(logits, dim=0)  # [m, n, p]
        C = F.softmax(logits.reshape(m, n * p), dim=1).reshape(m, n, p)  # [m, n, p]

        # Xs: [n, p, d]
        Xs = torch.einsum('md,mnp->npd', x_flat, D)

        Ys = torch.stack([self.experts[i](Xs[i]) for i in range(n)], dim=0)  # [n, p, d]

        # Y: [m, d]
        Y = torch.einsum('npd,mnp->md', Ys, C)

        base_out = self.base_expert(x_flat)
        # g = torch.sigmoid(self.raw_g)
        # print(g)
        final_out = Y + base_out

        return final_out.reshape(a, b, c)