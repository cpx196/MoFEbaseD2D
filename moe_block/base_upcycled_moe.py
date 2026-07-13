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

class B_U_MoE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
        moe_num_expert,
        moe_top_k,
        expert_choice: bool = False,
    ):
        super().__init__()
        self.register_buffer("expert_activation_count", torch.zeros(moe_num_expert, dtype=torch.long))
        self.num_expert = moe_num_expert
        self.d_model = hidden_dim
        self.k = moe_top_k
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward

        self.experts = nn.ModuleList(
            [_Expert(hidden_dim, dim_feedforward, dropout, activation) for _ in range(moe_num_expert)]
        )
        self.gate = NaiveGate(moe_num_expert, moe_top_k, hidden_dim)
        self.ec = expert_choice
        if self.ec:
            print("Using Expert Choice MoE")
        else:
            print("Using Token Choice MoE")

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

        for i, expert in enumerate(self.experts):
            # copy linear1
            if t_l1_w is not None:
                if tuple(t_l1_w.shape) != tuple(expert.linear1.weight.shape):
                    raise RuntimeError(f"shape mismatch for linear1.weight: ckpt {tuple(t_l1_w.shape)} vs expert {tuple(expert.linear1.weight.shape)}")
                expert.linear1.weight.data.copy_(t_l1_w.to(expert.linear1.weight.device))
            if t_l1_b is not None:
                if tuple(t_l1_b.shape) != tuple(expert.linear1.bias.shape):
                    raise RuntimeError(f"shape mismatch for linear1.bias: ckpt {tuple(t_l1_b.shape)} vs expert {tuple(expert.linear1.bias.shape)}")
                expert.linear1.bias.data.copy_(t_l1_b.to(expert.linear1.bias.device))

            # copy linear2
            if t_l2_w is not None:
                if tuple(t_l2_w.shape) != tuple(expert.linear2.weight.shape):
                    raise RuntimeError(f"shape mismatch for linear2.weight: ckpt {tuple(t_l2_w.shape)} vs expert {tuple(expert.linear2.weight.shape)}")
                expert.linear2.weight.data.copy_(t_l2_w.to(expert.linear2.weight.device))
            if t_l2_b is not None:
                if tuple(t_l2_b.shape) != tuple(expert.linear2.bias.shape):
                    raise RuntimeError(f"shape mismatch for linear2.bias: ckpt {tuple(t_l2_b.shape)} vs expert {tuple(expert.linear2.bias.shape)}")
                expert.linear2.bias.data.copy_(t_l2_b.to(expert.linear2.bias.device))

        print(f"init_experts_from_checkpoint: copied params to {len(self.experts)} experts from prefix '{prefix}'")

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

    def forward(self, x):
        a, b, c = x.shape
        x_flat = x.reshape(-1, c).contiguous()  # [tokens, c]

        if self.ec:
            # expert_choice MoE
            tokens = x_flat.size(0)
            num_expert = self.num_expert
            tokens_per_expert = (2 * tokens) // num_expert  # 平均每token被处理2次

            # gate 得分 shape: [tokens, num_expert]
            gate_score = self.gate.router(x_flat)   # [tokens, num_expert]
            gate_score = F.softmax(gate_score, dim=-1)  # softmax后分数
            gate_score_T = gate_score.transpose(0, 1)  # [num_expert, tokens]

            # 每个 expert 选 tokens_per_expert 个 token
            topk_scores, topk_indices = torch.topk(gate_score_T, tokens_per_expert, dim=1)  # [num_expert, tokens_per_expert]

            # 统计每个 token 被哪些 expert 选中，以及对应的门控分数
            token_expert_mask = torch.zeros(tokens, num_expert, dtype=torch.bool, device=x.device)
            token_expert_score = torch.zeros(tokens, num_expert, dtype=gate_score.dtype, device=x.device)
            for expert_idx in range(num_expert):
                token_idx = topk_indices[expert_idx]  # [tokens_per_expert]
                token_expert_mask[token_idx, expert_idx] = True
                token_expert_score[token_idx, expert_idx] = topk_scores[expert_idx]

            # 线性归一化：每个 token 被选中的 expert 的门控分数直接归一化
            score_sum = token_expert_score.sum(dim=1, keepdim=True)  # [tokens, 1]
            score_sum = score_sum + (score_sum == 0).float()  # 防止除零
            normed_score = token_expert_score / score_sum  # [tokens, num_expert]

            expert_outputs = torch.zeros_like(x_flat, device=x.device)

            for expert_idx, expert in enumerate(self.experts):
                token_idx = topk_indices[expert_idx]  # [tokens_per_expert]
                sel_inputs = x_flat[token_idx]        # [tokens_per_expert, c]
                out_i = expert(sel_inputs)            # [tokens_per_expert, c]
                w = normed_score[token_idx, expert_idx].unsqueeze(1)  # [tokens_per_expert, 1]
                expert_outputs[token_idx] += out_i * w

        else:
            # token choice MoE
            gate_scores, top_k_indices = self.gate(x)  # [tokens, k], [tokens, k]

            flat_indices = top_k_indices.view(-1)  # [tokens * k]
            counts = torch.bincount(flat_indices, minlength=self.num_expert)
            self.expert_activation_count += counts
            tokens = x_flat.size(0)

            # 1. 构造 (token_idx, expert_idx) 对
            token_idx = torch.arange(tokens, device=x.device).unsqueeze(1).expand(-1, self.k).reshape(-1)  # [tokens*k]
            expert_idx = top_k_indices.reshape(-1)  # [tokens*k]
            weights = gate_scores.reshape(-1, 1)    # [tokens*k, 1]

            # 2. 按 expert 分组，批量前向
            expert_outputs = torch.zeros_like(x_flat)
            for i, expert in enumerate(self.experts):
                mask = (expert_idx == i)
                if not mask.any():
                    continue
                sel_tokens = token_idx[mask]
                sel_inputs = x_flat[sel_tokens]  # [n_i, c]
                # expert = self.experts[5]
                out_i = expert(sel_inputs)       # [n_i, c]
                w = weights[mask]                # [n_i, 1]
                expert_outputs.index_add_(0, sel_tokens, out_i * w)

        return expert_outputs.reshape(a, b, c)