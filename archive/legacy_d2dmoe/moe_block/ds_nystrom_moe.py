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
    
class _Activate(nn.Module):
    def __init__(
        self,
        dropout,
        activation,
    ):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tgt = self.dropout(self.activation(x))
        return tgt

class DS_NA_MoE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
        moe_num_expert,
        moe_top_k,
        low_rank: int = 196,
        alpha: float = 0.5,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.num_expert = moe_num_expert
        self.d_model = hidden_dim
        self.k = moe_top_k
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.low_rank = low_rank
        self.freeze_base = freeze_base

        # self.experts = nn.ModuleList(
        #     [_Expert(hidden_dim, dim_feedforward, dropout, activation) for _ in range(moe_num_expert)]
        # )
        self.base_expert = _Expert(hidden_dim, dim_feedforward, dropout, activation)
        self.gate = NaiveGate(moe_num_expert, moe_top_k, hidden_dim)

        self.num = int(moe_num_expert ** 0.5)
        # low-rank adapters (A @ B) per expert for linear1 and linear2
        # linear1: out = dim_feedforward, in = hidden_dim
        nystrom_dim_feedforward = dim_feedforward - low_rank
        nystrom_hidden_dim = hidden_dim - low_rank
        self.lr1_A = nn.Parameter(torch.zeros(self.num, nystrom_dim_feedforward, low_rank))
        self.lr1_B = nn.Parameter(torch.zeros(self.num, low_rank, nystrom_hidden_dim))
        # linear2: out = hidden_dim, in = dim_feedforward
        self.lr2_A = nn.Parameter(torch.zeros(self.num, nystrom_hidden_dim, low_rank))
        self.lr2_B = nn.Parameter(torch.zeros(self.num, low_rank, nystrom_dim_feedforward))

        self.midmat_1 = nn.Parameter(torch.zeros(moe_num_expert, low_rank, low_rank))
        self.midmat_2 = nn.Parameter(torch.zeros(moe_num_expert, low_rank, low_rank))

        # optional bias deltas (initialized zero)
        self.lr1_bias = nn.Parameter(torch.zeros(moe_num_expert, dim_feedforward))
        self.lr2_bias = nn.Parameter(torch.zeros(moe_num_expert, hidden_dim))

        self.alpha = alpha

        self.activations = nn.ModuleList(
            [_Activate(dropout, activation) for _ in range(moe_num_expert)]
        )

        if self.freeze_base:
            for param in self.base_expert.parameters():
                param.requires_grad = False

        self._lora_initialized = False
        if not self._lora_initialized:
            self._reset_lora_parameters(init_scale=1e-3)

    def _reset_lora_parameters(self, init_scale: float = 1e-3):
        """
        Initialize LoRA parameters:
          - A matrices: random normal with std=init_scale
          - B matrices: zeros
          - bias deltas: zeros
        """
        print("Initializing LoRA parameters...")

        with torch.no_grad():
            nn.init.normal_(self.lr1_A, mean=0.0, std=init_scale)
            nn.init.normal_(self.lr2_A, mean=0.0, std=init_scale)
            # nn.init.zeros_(self.lr1_B)
            # nn.init.zeros_(self.lr2_B)
            nn.init.normal_(self.lr1_B, mean=0.0, std=init_scale)
            nn.init.normal_(self.lr2_B, mean=0.0, std=init_scale)
            nn.init.normal_(self.midmat_1, mean=0.0, std=init_scale)
            nn.init.normal_(self.midmat_2, mean=0.0, std=init_scale)

            nn.init.zeros_(self.lr1_bias)
            nn.init.zeros_(self.lr2_bias)

        self._lora_initialized = True

    # load params from base checkpoints
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
        k_l1_w = prefix + 'linear1.weight'
        k_l1_b = prefix + 'linear1.bias'
        k_l2_w = prefix + 'linear2.weight'
        k_l2_b = prefix + 'linear2.bias'

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

        for i in range (self.num_expert):
            self.midmat_1[i].data.copy_(self.base_expert.linear1.weight.data[:self.low_rank, :self.low_rank])
            self.midmat_2[i].data.copy_(self.base_expert.linear2.weight.data[:self.low_rank, :self.low_rank])
            self.lr1_bias[i].data.copy_(self.base_expert.linear1.bias.data)
            self.lr2_bias[i].data.copy_(self.base_expert.linear2.bias.data)
        
        for i in range (self.num):
            self.lr1_A[i].data.copy_(self.base_expert.linear1.weight.data[self.low_rank:, :self.low_rank])
            self.lr1_B[i].data.copy_(self.base_expert.linear1.weight.data[:self.low_rank, self.low_rank:])
            self.lr2_A[i].data.copy_(self.base_expert.linear2.weight.data[self.low_rank, :self.low_rank])
            self.lr2_B[i].data.copy_(self.base_expert.linear2.weight.data[:self.low_rank, self.low_rank:])

        # if self.freeze_base:
        #     for param in self.base_expert.parameters():
        #         param.requires_grad = False

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
            first_param = next(self.base_expert.parameters(), None)
            device = first_param.device if first_param is not None else torch.device("cpu")

        lr1_A = self.lr1_A.to(device)
        lr1_B = self.lr1_B.to(device)
        lr2_A = self.lr2_A.to(device)
        lr2_B = self.lr2_B.to(device)
        # delta shapes: [num_expert, out, in]
        # print(lr1_A)
        # print(lr1_B)

        # W1_base = self.base_expert.linear1.weight.to(device)  # [dim_feedforward, hidden_dim]
        # b1_base = self.base_expert.linear1.bias.to(device) if self.base_expert.linear1.bias is not None else None
        # W2_base = self.base_expert.linear2.weight.to(device)  # [dim_feedforward, hidden_dim]
        # b2_base = self.base_expert.linear2.bias.to(device) if self.base_expert.linear2.bias is not None else None

        vecs = []
        # for expert in self.experts:
        for i in range(self.num):
            for j in range(self.num):
                expert_idx = i * self.num + j
                if expert_idx >= self.num_expert:
                    continue
                parts = []
                # combine linear1
                W1_comb = torch.bmm(lr1_A[i].unsqueeze(0), lr1_B[j].unsqueeze(0)).squeeze(0)
                # combine linear2
                W2_comb = torch.bmm(lr2_A[i].unsqueeze(0), lr2_B[j].unsqueeze(0)).squeeze(0)


                pre_linear1_w = W1_comb
                pre_linear1_b = self.lr1_bias[i].to(device)
                pre_linear2_w = W2_comb
                pre_linear2_b = self.lr2_bias[i].to(device)

                if which in ("linear1", "both"):
                    parts.append(pre_linear1_w.detach().to(device).view(-1))
                    if include_bias and pre_linear1_b is not None:
                        parts.append(pre_linear1_b.detach().to(device).view(-1))
                if which in ("linear2", "both"):
                    parts.append(pre_linear2_w.detach().to(device).view(-1))
                    if include_bias and pre_linear2_b is not None:
                        parts.append(pre_linear2_b.detach().to(device).view(-1))

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

        # precompute low-rank deltas per expert via batch bmm
        device = x.device

        midmat_1_pinv = self.midmat_1.to(device).inverse()
        midmat_2_pinv = self.midmat_2.to(device).inverse()
        # midmat_1_pinv = torch.linalg.pinv(self.midmat_1.to(device))
        # midmat_2_pinv = torch.linalg.pinv(self.midmat_2.to(device))

        final_lr1_A = []
        final_lr1_B = []
        final_lr2_A = []
        final_lr2_B = []
        for i in range(self.num):
            for j in range(self.num):
                final_lr1_A.append(self.lr1_A[i].to(device))
                final_lr1_B.append(self.lr1_B[j].to(device))
                final_lr2_A.append(self.lr2_A[i].to(device))
                final_lr2_B.append(self.lr2_B[j].to(device))
        final_lr1_A = torch.stack(final_lr1_A, dim=0)  # [E, out, low_rank]
        final_lr1_B = torch.stack(final_lr1_B, dim=0)  # [E, low_rank, in]
        final_lr2_A = torch.stack(final_lr2_A, dim=0)  # [E, out, low_rank]
        final_lr2_B = torch.stack(final_lr2_B, dim=0)  #

        C_matrix1 = torch.cat([self.midmat_1, final_lr1_A], dim=1)  # [E, out, low_rank]
        T_matrix1 = torch.cat([self.midmat_1, final_lr1_B], dim=2)  # [E, low_rank, in]
        W1_comb = torch.bmm(C_matrix1, midmat_1_pinv)              # [E, out, low_rank]
        W1_comb = torch.bmm(W1_comb, T_matrix1)                    # [E, out, in]
        b1_comb = self.lr1_bias.to(device)

        C_matrix2 = torch.cat([self.midmat_2, final_lr2_A], dim=1)
        T_matrix2 = torch.cat([self.midmat_2, final_lr2_B], dim=2)
        W2_comb = torch.bmm(C_matrix2, midmat_2_pinv)
        W2_comb = torch.bmm(W2_comb, T_matrix2)
        b2_comb = self.lr2_bias.to(device)

        # W1_base = self.base_expert.linear1.weight.to(device)  # [dim_feedforward, hidden_dim]
        # b1_base = self.base_expert.linear1.bias.to(device) if self.base_expert.linear1.bias is not None else None
        # W2_base = self.base_expert.linear2.weight.to(device)  # [hidden_dim, dim_feedforward]
        # b2_base = self.base_expert.linear2.bias.to(device) if self.base_expert.linear2.bias is not None else None

        for i in range(self.num_expert):
            expert_idx = i
            if expert_idx >= self.num_expert:
                continue
            mask = masks_matrix[:, expert_idx]
            if not mask.any():
                continue
            selected_inputs = x_flat[mask]          # [n_i, c]

            # W1_pre = W1_base + W1_comb[expert_idx]
            # W2_pre = W2_base + W2_comb[expert_idx]
            # b1_pre = b1_base + b1_comb[expert_idx] if b1_base is not None else None
            # b2_pre = b2_base + b2_comb[expert_idx] if b2_base is not None else None

            t = F.linear(selected_inputs, W1_comb[expert_idx], b1_comb[expert_idx])
            t = self.activations[expert_idx](t)

            out_i = F.linear(t, W2_comb[expert_idx], b2_comb[expert_idx])         # [n_i, c]
            # apply per-token weight for this expert
            w = weights_per_token[mask, expert_idx].unsqueeze(1)  # [n_i, 1]
            out_base = self.base_expert(selected_inputs)  
            expert_outputs[mask] += out_i * w + out_base

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