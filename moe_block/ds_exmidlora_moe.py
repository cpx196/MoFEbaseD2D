# ...existing code...
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
        return gate_score

class NoisyGate(nn.Module):
    def __init__(self, num_experts, k, d_model, num_expert):
        super().__init__()
        self.num = num_experts
        self.k = k
        self.noise_std = 1.0 / num_expert
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Accept x with shape [seq_len, batch, d_model] or [tokens, d_model]
        if x.dim() == 3:
            seq, b, d = x.shape
            x_flat = x.reshape(-1, d)
        else:
            x_flat = x

        gate_score = self.router(x_flat)  # [tokens, num_expert]
        if self.training:
            noise = torch.randn_like(gate_score) * self.noise_std
            gate_score = gate_score + noise
        return gate_score

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

class DS_EX_MidMt_MoE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
        moe_num_expert,
        moe_top_k,
        expert_choice: bool = True,
        low_rank_ratio: float = 0.75,
        freeze_base: bool = False,
        freeze_lora: bool = False,
    ):
        super().__init__()
        self.register_buffer("expert_activation_count", torch.zeros(moe_num_expert, dtype=torch.long))
        self.num_expert = moe_num_expert
        self.d_model = hidden_dim
        self.k = moe_top_k
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.low_rank = int(low_rank_ratio * hidden_dim)
        self.freeze_base = freeze_base
        self.freeze_lora = freeze_lora

        self.share = True

        self.base_expert = _Expert(hidden_dim, dim_feedforward, dropout, activation)
        self.gate = NaiveGate(moe_num_expert, moe_top_k, hidden_dim)
        # self.gate = NoisyGate(moe_num_expert, moe_top_k, hidden_dim, self.num_expert)

        if self.share:
            self.num = int(moe_num_expert ** 0.5)
        else:
            self.num = moe_num_expert

        self.lr1_A = nn.Parameter(torch.zeros(self.num, dim_feedforward, self.low_rank))
        self.lr1_B = nn.Parameter(torch.zeros(self.num, self.low_rank, hidden_dim))
        # linear2: out = hidden_dim, in = dim_feedforward
        self.lr2_A = nn.Parameter(torch.zeros(self.num, hidden_dim, self.low_rank))
        self.lr2_B = nn.Parameter(torch.zeros(self.num, self.low_rank, dim_feedforward))

        self.midmat_1 = nn.Parameter(torch.zeros(moe_num_expert, self.low_rank, self.low_rank))
        self.midmat_2 = nn.Parameter(torch.zeros(moe_num_expert, self.low_rank, self.low_rank))

        # optional bias deltas (initialized zero)
        self.lr1_bias = nn.Parameter(torch.zeros(moe_num_expert, dim_feedforward))
        self.lr2_bias = nn.Parameter(torch.zeros(moe_num_expert, hidden_dim))

        self.activations = nn.ModuleList(
            [_Activate(dropout, activation) for _ in range(moe_num_expert)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(moe_num_expert)]
        )

        if self.freeze_base:
            print("Freeze the base expert!")
            for param in self.base_expert.parameters():
                param.requires_grad = False

        if self.freeze_lora:
            print("Lora matrix frozen")
            self.lr1_A.requires_grad = False
            self.lr1_B.requires_grad = False
            self.lr2_A.requires_grad = False
            self.lr2_B.requires_grad = False

        self.ec = expert_choice
        if self.ec:
            print("Expert choice MoE")  
        else:
            print("Token choice MoE")

        self._lora_initialized = False
        init_scale = float(1e-1 * 192 / hidden_dim)
        print(f"DS_EX_MidMt_MoE init: low_rank={self.low_rank}, init_scale={init_scale:.4f}, hidden_dim={hidden_dim}")
        if not self._lora_initialized:
            self._reset_lora_parameters(init_scale=init_scale)

    def _reset_lora_parameters(self, init_scale: float = 1e-1):
        """
        Initialize LoRA parameters:
          - A matrices: random normal with std=init_scale
          - B matrices: zeros
          - bias deltas: zeros
        """
        print("Initializing LoRA parameters...")

        nn.init.normal_(self.lr1_A, mean=0.0, std=init_scale)
        nn.init.normal_(self.lr2_A, mean=0.0, std=init_scale)
        # nn.init.zeros_(self.lr1_B)
        # nn.init.zeros_(self.lr2_B)
        nn.init.normal_(self.lr1_B, mean=0.0, std=init_scale)
        nn.init.normal_(self.lr2_B, mean=0.0, std=init_scale)
        nn.init.normal_(self.midmat_1, mean=0.0, std=init_scale)
        nn.init.normal_(self.midmat_2, mean=0.0, std=init_scale)
        # nn.init.zeros_(self.midmat_1)
        # nn.init.zeros_(self.midmat_2)            
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
        k_l1_w = prefix + 'mlp.fc1.weight'
        k_l1_b = prefix + 'mlp.fc1.bias'
        k_l2_w = prefix + 'mlp.fc2.weight'
        k_l2_b = prefix + 'mlp.fc2.bias'

        print(f"init_experts_from_checkpoint: looking for keys '{k_l1_w}', '{k_l1_b}', '{k_l2_w}', '{k_l2_b}'")

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

        print("Nystrom init matrix A and B")
        for i in range (self.num):
            self.lr1_A[i].data.copy_(self.base_expert.linear1.weight.data[:, :self.low_rank])
            self.lr1_B[i].data.copy_(self.base_expert.linear1.weight.data[:self.low_rank, :])
            self.lr2_A[i].data.copy_(self.base_expert.linear2.weight.data[:, :self.low_rank])
            self.lr2_B[i].data.copy_(self.base_expert.linear2.weight.data[:self.low_rank, :])

        # for i in range (self.num_expert):
        #     print(f"Before init, midmat_1[{i}] norm: {self.midmat_1[i].norm().item():.4f}, midmat_2[{i}] norm: {self.midmat_2[i].norm().item():.4f}")
        #     device = self.midmat_1.device
        #     base_weight_1 = self.base_expert.linear1.weight.data[:self.low_rank, :self.low_rank].to(device)
        #     base_weight_2 = self.base_expert.linear2.weight.data[:self.low_rank, :self.low_rank].to(device)
        #     self.midmat_1.data.add_(base_weight_1.unsqueeze(0))
        #     self.midmat_2.data.add_(base_weight_2.unsqueeze(0))

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
        detach: bool = True,
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
        # delta1 = torch.bmm(lr1_A, lr1_B)  # [E, dim_feedforward, hidden_dim]
        # delta2 = torch.bmm(lr2_A, lr2_B)  # [E, hidden_dim, dim_feedforward]

        # print(lr1_A)
        # print(lr1_B)

        W1_base = self.base_expert.linear1.weight.to(device)  # [dim_feedforward, hidden_dim]
        b1_base = self.base_expert.linear1.bias.to(device) if self.base_expert.linear1.bias is not None else None
        W2_base = self.base_expert.linear2.weight.to(device)  # [dim_feedforward, hidden_dim]
        b2_base = self.base_expert.linear2.bias.to(device) if self.base_expert.linear2.bias is not None else None
        midmat_1 = self.midmat_1.to(device)
        midmat_2 = self.midmat_2.to(device)

        vecs = []
        # for expert in self.experts:
        if self.share:
            for i in range(self.num):
                for j in range(self.num):
                    expert_idx = i * self.num + j
                    if expert_idx >= self.num_expert:
                        continue
                    parts = []
                    # combine linear1
                    delta_A1 = torch.bmm(lr1_A[i].unsqueeze(0), midmat_1[expert_idx].unsqueeze(0)).squeeze(0)  # [dim_feedforward, low_rank]
                    delta_B1 = torch.bmm(delta_A1.unsqueeze(0), lr1_B[j].unsqueeze(0)).squeeze(0)  # [dim_feedforward, hidden_dim]
                    W1_comb = W1_base + delta_B1
                    # combine linear2
                    delta_A2 = torch.bmm(lr2_A[i].unsqueeze(0), midmat_2[expert_idx].unsqueeze(0)).squeeze(0)  # [hidden_dim, low_rank]
                    delta_B2 = torch.bmm(delta_A2.unsqueeze(0), lr2_B[j].unsqueeze(0)).squeeze(0)  # [hidden_dim, dim_feedforward]
                    W2_comb = W2_base + delta_B2

                    pre_linear1_w = W1_comb
                    pre_linear1_b = (b1_base + self.lr1_bias[i].to(device)) if b1_base is not None else None
                    pre_linear2_w = W2_comb
                    pre_linear2_b = (b2_base + self.lr2_bias[i].to(device)) if b2_base is not None else None

                    # print("Expert", expert_idx, "linear1 weight:", pre_linear1_w)

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
        
        else:
            for i in range(self.num):
                expert_idx = i
                if expert_idx >= self.num_expert:
                    continue
                parts = []
                # combine linear1
                delta_A1 = torch.bmm(lr1_A[i].unsqueeze(0), midmat_1[expert_idx].unsqueeze(0)).squeeze(0)  # [dim_feedforward, low_rank]
                delta_B1 = torch.bmm(delta_A1.unsqueeze(0), lr1_B[i].unsqueeze(0)).squeeze(0)  # [dim_feedforward, hidden_dim]
                W1_comb = W1_base + delta_B1
                # combine linear2
                delta_A2 = torch.bmm(lr2_A[i].unsqueeze(0), midmat_2[expert_idx].unsqueeze(0)).squeeze(0)  # [hidden_dim, low_rank]
                delta_B2 = torch.bmm(delta_A2.unsqueeze(0), lr2_B[i].unsqueeze(0)).squeeze(0)  # [hidden_dim, dim_feedforward]
                W2_comb = W2_base + delta_B2

                pre_linear1_w = W1_comb
                pre_linear1_b = (b1_base + self.lr1_bias[i].to(device)) if b1_base is not None else None
                pre_linear2_w = W2_comb
                pre_linear2_b = (b2_base + self.lr2_bias[i].to(device)) if b2_base is not None else None

                # print("Expert", expert_idx, "linear1 weight:", pre_linear1_w)

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

    def forward(self, x):
        a, b, c = x.shape
        x_cls = x[:, 0, :]  # [batch, d_model]
        x_flat = x.reshape(-1, c).contiguous()  # [tokens, d_model]

        if self.ec:
            tokens = x_flat.size(0)
            num_expert = self.num_expert
            tokens_per_expert = (2 * tokens) // num_expert  # 平均每token被处理2次

            # gate 得分 shape: [tokens, num_expert]
            gate_score = self.gate(x_flat)   # [tokens, num_expert]
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

            # print(token_expert_mask)
            # 线性归一化：每个 token 被选中的 expert 的门控分数直接归一化
            score_sum = token_expert_score.sum(dim=1, keepdim=True)  # [tokens, 1]
            # # 防止除零
            # score_sum = score_sum + (score_sum == 0).float()
            score_sum = score_sum + 1  # [tokens, 1]
            normed_score = token_expert_score / score_sum  # [tokens, num_expert]

            expert_outputs = torch.zeros_like(x_flat, device=x.device)

            # LoRA权重等准备与原代码一致
            device = x.device
            midmat_1 = self.midmat_1.to(device)
            midmat_2 = self.midmat_2.to(device)
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
            final_lr2_B = torch.stack(final_lr2_B, dim=0)
            delta_A1 = torch.bmm(final_lr1_A, midmat_1)
            delta_B1 = torch.bmm(delta_A1, final_lr1_B)
            delta_A2 = torch.bmm(final_lr2_A, midmat_2)
            delta_B2 = torch.bmm(delta_A2, final_lr2_B)
            lr1_bias = self.lr1_bias.to(device)
            lr2_bias = self.lr2_bias.to(device)

            for expert_idx in range(num_expert):
                token_idx = topk_indices[expert_idx]  # [tokens_per_expert]
                selected_inputs = x_flat[token_idx]   # [tokens_per_expert, d_model]

                idx = expert_idx
                # LoRA权重组合
                W1_comb = delta_B1[idx]
                b1_comb = lr1_bias[idx]
                t = F.linear(selected_inputs, W1_comb, b1_comb)
                t = self.activations[idx](t)
                W2_comb = delta_B2[idx]
                b2_comb = lr2_bias[idx]
                out_i = F.linear(t, W2_comb, b2_comb)  # [tokens_per_expert, d_model]

                # 用线性归一化后的门控分数加权
                w = normed_score[token_idx, expert_idx].unsqueeze(1)  # [tokens_per_expert, 1]
                # out_b = self.base_expert(selected_inputs)  # [tokens_per_expert, d_model]
                
                # w_b = score_sum[token_idx, 0].unsqueeze(1)
                expert_outputs[token_idx] += out_i * w

            base_out = self.base_expert(x_flat)
            w_base = 1 / score_sum
            final_out = expert_outputs + base_out * w_base
            # final_out = expert_outputs + expert_outputs

        else:
            # gate_score_cls = self.gate(x_cls)   # [batch, num_expert]
            # gate_score_cls = gate_score_cls.unsqueeze(1).expand(-1, b, -1).reshape(-1, self.num_expert)
            gate_score = self.gate(x_flat)   # [tokens, num_expert]
            topk_scores, topk_indices = torch.topk(gate_score, self.k, dim=-1)  # [tokens, k]
            # topk_scores, topk_indices = torch.topk(gate_score_cls, self.k, dim=-1)  # [tokens, k]

            topk_scores = topk_scores.view(-1, self.k)
            top_k_indices = topk_indices.view(-1, self.k)
            gate_scores = F.softmax(topk_scores, dim=-1)

            flat_indices = top_k_indices.view(-1)  # [tokens * k]
            counts = torch.bincount(flat_indices, minlength=self.num_expert)
            self.expert_activation_count += counts

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

            midmat_1 = self.midmat_1.to(device)
            midmat_2 = self.midmat_2.to(device)

            final_lr1_A = []
            final_lr1_B = []
            final_lr2_A = []
            final_lr2_B = []

            if self.share:
                for i in range(self.num):
                    for j in range(self.num):
                        final_lr1_A.append(self.lr1_A[i].to(device))
                        final_lr1_B.append(self.lr1_B[j].to(device))
                        final_lr2_A.append(self.lr2_A[i].to(device))
                        final_lr2_B.append(self.lr2_B[j].to(device))
            else:
                for i in range(self.num):
                    final_lr1_A.append(self.lr1_A[i].to(device))
                    final_lr1_B.append(self.lr1_B[i].to(device))
                    final_lr2_A.append(self.lr2_A[i].to(device))
                    final_lr2_B.append(self.lr2_B[i].to(device))
            final_lr1_A = torch.stack(final_lr1_A, dim=0)  # [E, out, low_rank]
            final_lr1_B = torch.stack(final_lr1_B, dim=0)  # [E, low_rank, in]
            final_lr2_A = torch.stack(final_lr2_A, dim=0)  # [E, out, low_rank]
            final_lr2_B = torch.stack(final_lr2_B, dim=0)  #
            delta_A1 = torch.bmm(final_lr1_A, midmat_1)
            delta_B1 = torch.bmm(delta_A1, final_lr1_B)
            delta_A2 = torch.bmm(final_lr2_A, midmat_2)
            delta_B2 = torch.bmm(delta_A2, final_lr2_B)

            lr1_bias = self.lr1_bias.to(device)
            lr2_bias = self.lr2_bias.to(device)

            for expert_idx in range(self.num_expert):
                if expert_idx >= self.num_expert:
                    continue
                mask = masks_matrix[:, expert_idx]
                if not mask.any():
                    continue
                selected_inputs = x_flat[mask]          # [n_i, c]

                W1_comb = delta_B1[expert_idx]
                b1_comb = lr1_bias[expert_idx]

                t = F.linear(selected_inputs, W1_comb, b1_comb)
                t = self.dropouts[expert_idx](self.activations[expert_idx](t))

                W2_comb = delta_B2[expert_idx]
                b2_comb = lr2_bias[expert_idx]

                out_i = F.linear(t, W2_comb, b2_comb)         # [n_i, c]
                # apply per-token weight for this expert
                w = weights_per_token[mask, expert_idx].unsqueeze(1)  # [n_i, 1]
                expert_outputs[mask] += out_i * w

            base_out = self.base_expert(x_flat)
            final_out = expert_outputs + base_out
            # final_out = expert_outputs

        return final_out.reshape(a, b, c)

