# ...existing code...
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
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

class NaiveGate(nn.Module):
    def __init__(self, num_experts, k, d_model):
        super().__init__()
        self.num = num_experts
        self.k = k
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Accept x with shape [seq_len, batch, d_model] or [tokens, d_model]
        if x.dim() == 3:
            seq, b, d = x.shape
            x_flat = x.reshape(-1, d)
        else:
            x_flat = x

        gate_score = self.router(x_flat)  # [tokens, num_expert]
        topk_scores, topk_indices = torch.topk(gate_score, self.k, dim=-1)  # [tokens, k]

        # keep same flattening behavior as original: return [tokens, k]
        topk_scores = topk_scores.view(-1, self.k)
        topk_indices = topk_indices.view(-1, self.k)
        output_score = F.softmax(topk_scores, dim=-1)

        return output_score, topk_indices

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

class V_MoE(nn.Module):
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
        self.d_model = hidden_dim
        self.k = moe_top_k
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward

        self.experts = nn.ModuleList(
            [_Expert(hidden_dim, dim_feedforward, dropout, activation) for _ in range(moe_num_expert)]
        )
        self.gate = NaiveGate(moe_num_expert, moe_top_k, hidden_dim)

    # 新增：比较 experts 之间参数的余弦相似度
    def experts_cosine_similarity(
        self,
        which: str = "both",
        include_bias: bool = False,
        device: str = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        计算专家之间参数向量的两两余弦相似度矩阵并返回 (num_expert, num_expert) 的 tensor。

        参数:
          which: 'linear1' | 'linear2' | 'both'（默认） - 选择用哪部分参数来比较
          include_bias: 是否把对应线性层的 bias 一并拼接进向量(默认 False)
          device: 结果 tensor 的 device(默认使用专家参数所在 device)
          eps: 用于 normalize 的小常数

        返回:
          sim: shape [num_expert, num_expert] 的余弦相似度矩阵，值域在 [-1, 1]。
        """
        if which not in ("linear1", "linear2", "both"):
            raise ValueError("which must be 'linear1', 'linear2' or 'both'")

        # choose device
        if device is None:
            # use first expert param device
            first_param = next(self.experts[0].parameters(), None)
            device = first_param.device if first_param is not None else torch.device("cpu")

        vecs = []
        for expert in self.experts:
            parts = []
            if which in ("linear1", "both"):
                parts.append(expert.linear1.weight.detach().to(device).view(-1))
                if include_bias and expert.linear1.bias is not None:
                    parts.append(expert.linear1.bias.detach().to(device).view(-1))
            if which in ("linear2", "both"):
                parts.append(expert.linear2.weight.detach().to(device).view(-1))
                if include_bias and expert.linear2.bias is not None:
                    parts.append(expert.linear2.bias.detach().to(device).view(-1))

            if not parts:
                raise RuntimeError("No parameters selected for comparison.")
            vec = torch.cat(parts, dim=0)
            vecs.append(vec)

        # stack -> [num_expert, dim]
        mat = torch.stack(vecs, dim=0)  # detach already called
        # normalize per-row
        mat = F.normalize(mat, p=2, dim=1, eps=eps)
        sim = mat @ mat.t()  # cosine similarity matrix
        return sim

    # # ...existing code...
    def forward(self, x):
        a, b, c = x.shape
        x_flat = x.reshape(-1, c).contiguous()  # tokens x c
        gate_scores, top_k_indices = self.gate(x)  # gate_scores: [tokens, k], top_k_indices: [tokens, k]

        tokens = x_flat.size(0)

        # vectorized: build per-token-per-expert weight (sparse by k choices)
        weights_per_token = torch.zeros(tokens, self.num_expert, device=x.device, dtype=gate_scores.dtype)
        # small loop over k (k is small, e.g. 2) is fine
        for kk in range(self.k):
            idx = top_k_indices[:, kk].unsqueeze(1)           # [tokens,1]
            weights_per_token.scatter_add_(1, idx, gate_scores[:, kk].unsqueeze(1))  # accumulate weights

        # vectorized masks: tokens x num_expert boolean (True if expert selected by any of top-k)
        # shape broadcasting: top_k_indices.unsqueeze(-1): [tokens, k, 1]
        expert_ids = torch.arange(self.num_expert, device=x.device).view(1, 1, -1)  # [1,1,num_expert]
        masks_matrix = (top_k_indices.unsqueeze(-1) == expert_ids).any(dim=1)     # [tokens, num_expert]

        expert_outputs = torch.zeros_like(x_flat, device=x.device)  # [tokens, c]
        # per-expert processing loop (still necessary because each expert has different params)
        for i, expert in enumerate(self.experts):
            mask = masks_matrix[:, i]
            if not mask.any():
                continue
            selected_inputs = x_flat[mask]          # [n_i, c]
            out_i = expert(selected_inputs)         # [n_i, c]
            # apply per-token weight for this expert
            w = weights_per_token[mask, i].unsqueeze(1)  # [n_i, 1]
            expert_outputs[mask] += out_i * w

        return expert_outputs.reshape(a, b, c)
# ...existing code...

# # optional quick test
# if __name__ == "__main__":
#     seq = 196
#     bsz = 8
#     hidden = 256
#     ff = 512
#     moex = 8
#     topk = 2
#     model = V_MoE(hidden, ff, 0.1, "relu", moex, topk).cuda()
#     x = torch.randn(seq, bsz, hidden).cuda()
#     out = model(x)
#     print("out.shape:", out.shape)
# # ...existing code...