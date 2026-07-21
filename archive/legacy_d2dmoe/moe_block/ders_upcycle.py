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

class SparseExpertMatrix(nn.Module):
    def __init__(self, vector_dim, matrix_size, p):
        super().__init__()
        self.vector_dim = vector_dim
        self.matrix_size = matrix_size
        self.p = p

        total_size = matrix_size * vector_dim
        num_nonzero = max(1, int(total_size * (1 - p)))

        # 随机选取 num_nonzero 个唯一索引
        chosen_indices = torch.randperm(total_size)[:num_nonzero]
        self.register_buffer("chosen_indices", chosen_indices)

        # 只保存非零参数
        self.value_vector = nn.Parameter(torch.zeros(num_nonzero))

    def forward(self):
        # 构造稀疏矩阵
        sparse_matrix = torch.zeros(self.matrix_size * self.vector_dim, device=self.value_vector.device)
        sparse_matrix.scatter_(0, self.chosen_indices, self.value_vector)
        sparse_matrix = sparse_matrix.view(self.matrix_size, self.vector_dim)
        return sparse_matrix


class DeRS_MoE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
        moe_num_expert,
        moe_top_k,
        low_rank: int = 400,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.num_expert = moe_num_expert
        self.d_model = hidden_dim
        self.k = moe_top_k
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.low_rank = low_rank
        self.freeze_base = freeze_base

        self.base_expert = _Expert(hidden_dim, dim_feedforward, dropout, activation)
        self.gate = NaiveGate(moe_num_expert, moe_top_k, hidden_dim)

        # # low-rank adapters (A @ B) per expert for linear1 and linear2
        # # linear1: out = dim_feedforward, in = hidden_dim
        # self.lr1_A = nn.Parameter(torch.zeros(moe_num_expert, dim_feedforward, low_rank))
        # self.lr1_B = nn.Parameter(torch.zeros(moe_num_expert, low_rank, hidden_dim))
        # # linear2: out = hidden_dim, in = dim_feedforward
        # self.lr2_A = nn.Parameter(torch.zeros(moe_num_expert, hidden_dim, low_rank))
        # self.lr2_B = nn.Parameter(torch.zeros(moe_num_expert, low_rank, dim_feedforward))

        # optional bias deltas (initialized zero)
        self.lr1_bias = nn.Parameter(torch.zeros(moe_num_expert, dim_feedforward))
        self.lr2_bias = nn.Parameter(torch.zeros(moe_num_expert, hidden_dim))

        self.activations = nn.ModuleList(
            [_Activate(dropout, activation) for _ in range(moe_num_expert)]
        )

        self.sparse_expert_matrices_1 = nn.ModuleList([
            SparseExpertMatrix(
                vector_dim=hidden_dim,
                matrix_size=dim_feedforward,
                p=0.9
            ) for _ in range(moe_num_expert)
        ])

        self.sparse_expert_matrices_2 = nn.ModuleList([
            SparseExpertMatrix(
                vector_dim=dim_feedforward,
                matrix_size=hidden_dim,
                p=0.9
            ) for _ in range(moe_num_expert)
        ])

        if self.freeze_base:
            for param in self.base_expert.parameters():
                param.requires_grad = False

        self._lora_initialized = False
        # if not self._lora_initialized:
        #     self._reset_lora_parameters(init_scale=1e-3)

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

        # lr1_A = self.lr1_A.to(device)
        # lr1_B = self.lr1_B.to(device)
        # lr2_A = self.lr2_A.to(device)
        # lr2_B = self.lr2_B.to(device)
        # # delta shapes: [num_expert, out, in]
        # delta1 = torch.bmm(lr1_A, lr1_B)  # [E, dim_feedforward, hidden_dim]
        # delta2 = torch.bmm(lr2_A, lr2_B)  # [E, hidden_dim, dim_feedforward]

        all_sparse_matrices_1 = [m() for m in self.sparse_expert_matrices_1]  # 列表长度为moe_num_expert
        all_sparse_matrices_2 = [m() for m in self.sparse_expert_matrices_2]  # 列表长度为moe_num_expert

        # print(lr1_A)
        # print(lr1_B)

        W1_base = self.base_expert.linear1.weight.to(device)  # [dim_feedforward, hidden_dim]
        b1_base = self.base_expert.linear1.bias.to(device) if self.base_expert.linear1.bias is not None else None
        W2_base = self.base_expert.linear2.weight.to(device)  # [dim_feedforward, hidden_dim]
        b2_base = self.base_expert.linear2.bias.to(device) if self.base_expert.linear2.bias is not None else None

        vecs = []
        # for expert in self.experts:
        for i in range(self.num_expert):
            parts = []
            pre_linear1_w = W1_base + all_sparse_matrices_1[i]
            pre_linear1_b = (b1_base + self.lr1_bias[i].to(device)) if b1_base is not None else None
            pre_linear2_w = W2_base + all_sparse_matrices_2[i]
            pre_linear2_b = (b2_base + self.lr2_bias[i].to(device)) if b2_base is not None else None

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

        # # precompute low-rank deltas per expert via batch bmm
        device = x.device
        # lr1_A = self.lr1_A.to(device)
        # lr1_B = self.lr1_B.to(device)
        # lr2_A = self.lr2_A.to(device)
        # lr2_B = self.lr2_B.to(device)
        # # delta shapes: [num_expert, out, in]
        # delta1 = torch.bmm(lr1_A, lr1_B)  # [E, dim_feedforward, hidden_dim]
        # delta2 = torch.bmm(lr2_A, lr2_B)  # [E, hidden_dim, dim_feedforward]

        all_sparse_matrices_1 = [m() for m in self.sparse_expert_matrices_1]  # 列表长度为moe_num_expert
        all_sparse_matrices_2 = [m() for m in self.sparse_expert_matrices_2]  # 列表长度为moe_num_expert
        # print(delta1.size())
        # print(all_sparse_matrices_1[0].size())
        # print(all_sparse_matrices_2[0].size())

        # per-expert processing loop (still necessary because each expert has different params)
        # for i, expert in enumerate(self.experts):
        for i in range(self.num_expert):
            mask = masks_matrix[:, i]
            if not mask.any():
                continue
            selected_inputs = x_flat[mask]          # [n_i, c]

            # combine linear1
            W1_base = self.base_expert.linear1.weight.to(device)  # [dim_feedforward, hidden_dim]
            b1_base = self.base_expert.linear1.bias.to(device) if self.base_expert.linear1.bias is not None else None

            # use sparse matrix
            W1_comb = W1_base + all_sparse_matrices_1[i]
            b1_comb = (b1_base + self.lr1_bias[i].to(device)) if b1_base is not None else None

            t = F.linear(selected_inputs, W1_comb, b1_comb)
            t = self.activations[i](t)

            # combine linear2
            W2_base = self.base_expert.linear2.weight.to(device)  # [hidden_dim, dim_feedforward]
            b2_base = self.base_expert.linear2.bias.to(device) if self.base_expert.linear2.bias is not None else None
            W2_comb = W2_base + all_sparse_matrices_2[i]
            b2_comb = (b2_base + self.lr2_bias[i].to(device)) if b2_base is not None else None

            out_i = F.linear(t, W2_comb, b2_comb)         # [n_i, c]
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