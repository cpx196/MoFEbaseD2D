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

class Soft_EX_MidMt_MoE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
        moe_num_expert,
        moe_top_k,
        low_rank: int = 196,
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
        self.low_rank = low_rank
        self.freeze_base = freeze_base
        self.freeze_lora = freeze_lora

        self.base_expert = _Expert(hidden_dim, dim_feedforward, dropout, activation)
        # self.gate = NaiveGate(moe_num_expert, moe_top_k, hidden_dim)

        self.num = int(moe_num_expert ** 0.5)

        self.lr1_A = nn.Parameter(torch.zeros(self.num, dim_feedforward, low_rank))
        self.lr1_B = nn.Parameter(torch.zeros(self.num, low_rank, hidden_dim))
        # linear2: out = hidden_dim, in = dim_feedforward
        self.lr2_A = nn.Parameter(torch.zeros(self.num, hidden_dim, low_rank))
        self.lr2_B = nn.Parameter(torch.zeros(self.num, low_rank, dim_feedforward))

        self.midmat_1 = nn.Parameter(torch.zeros(moe_num_expert, low_rank, low_rank))
        self.midmat_2 = nn.Parameter(torch.zeros(moe_num_expert, low_rank, low_rank))

        # optional bias deltas (initialized zero)
        self.lr1_bias = nn.Parameter(torch.zeros(moe_num_expert, dim_feedforward))
        self.lr2_bias = nn.Parameter(torch.zeros(moe_num_expert, hidden_dim))

        self.p = 16
        self.Phi = nn.Parameter(torch.zeros(hidden_dim, moe_num_expert, self.p)) # (d, n, p)
        nn.init.normal_(self.Phi, mean=0.0, std=1e-3)

        self.activations = nn.ModuleList(
            [_Activate(dropout, activation) for _ in range(moe_num_expert)]
        )

        if self.freeze_base:
            for param in self.base_expert.parameters():
                param.requires_grad = False

        if self.freeze_lora:
            print("Lora matrix frozen")
            self.lr1_A.requires_grad = False
            self.lr1_B.requires_grad = False
            self.lr2_A.requires_grad = False
            self.lr2_B.requires_grad = False

        self._lora_initialized = False
        if not self._lora_initialized:
            self._reset_lora_parameters(init_scale=1e-1)

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

        for i in range (self.num):
            self.lr1_A[i].data.copy_(self.base_expert.linear1.weight.data[:, :self.low_rank])
            self.lr1_B[i].data.copy_(self.base_expert.linear1.weight.data[:self.low_rank, :])
            self.lr2_A[i].data.copy_(self.base_expert.linear2.weight.data[:, :self.low_rank])
            self.lr2_B[i].data.copy_(self.base_expert.linear2.weight.data[:self.low_rank, :])
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

        W1_base = self.base_expert.linear1.weight.to(device)  # [dim_feedforward, hidden_dim]
        b1_base = self.base_expert.linear1.bias.to(device) if self.base_expert.linear1.bias is not None else None
        W2_base = self.base_expert.linear2.weight.to(device)  # [dim_feedforward, hidden_dim]
        b2_base = self.base_expert.linear2.bias.to(device) if self.base_expert.linear2.bias is not None else None
        midmat_1 = self.midmat_1.to(device)
        midmat_2 = self.midmat_2.to(device)

        vecs = []
        # for expert in self.experts:
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

        # stack -> [num_expert, dim]
        mat = torch.stack(vecs, dim=0)  # detach already called
        # normalize per-row
        mat = F.normalize(mat, p=2, dim=1, eps=eps)
        sim = mat @ mat.t()  # cosine similarity matrix
        return sim

    # def experts_generate(self, device):
    #     midmat_1 = self.midmat_1.to(device)
    #     midmat_2 = self.midmat_2.to(device)
    #     final_lr1_A = []
    #     final_lr1_B = []
    #     final_lr2_A = []
    #     final_lr2_B = []
    #     for i in range(self.num):
    #         for j in range(self.num):
    #             final_lr1_A.append(self.lr1_A[i].to(device))
    #             final_lr1_B.append(self.lr1_B[j].to(device))
    #             final_lr2_A.append(self.lr2_A[i].to(device))
    #             final_lr2_B.append(self.lr2_B[j].to(device))
    #     final_lr1_A = torch.stack(final_lr1_A, dim=0)  # [E, out, low_rank]
    #     final_lr1_B = torch.stack(final_lr1_B, dim=0)  # [E, low_rank, in]
    #     final_lr2_A = torch.stack(final_lr2_A, dim=0)  # [E, out, low_rank]
    #     final_lr2_B = torch.stack(final_lr2_B, dim=0)
    #     delta_A1 = torch.bmm(final_lr1_A, midmat_1)
    #     delta_B1 = torch.bmm(delta_A1, final_lr1_B)
    #     delta_A2 = torch.bmm(final_lr2_A, midmat_2)
    #     delta_B2 = torch.bmm(delta_A2, final_lr2_B)
    #     return delta_B1, delta_B2

    def experts_generate(self, device):
        # 原参数
        lr1_A = self.lr1_A.to(device)  # [num, dim_feedforward, low_rank]
        lr1_B = self.lr1_B.to(device)  # [num, low_rank, hidden_dim]
        lr2_A = self.lr2_A.to(device)  # [num, hidden_dim, low_rank]
        lr2_B = self.lr2_B.to(device)  # [num, low_rank, dim_feedforward]
        midmat_1 = self.midmat_1.to(device)  # [num_expert, low_rank, low_rank]
        midmat_2 = self.midmat_2.to(device)  # [num_expert, low_rank, low_rank]

        # meshgrid生成所有(i, j)组合
        num = self.num
        num_expert = self.num_expert
        idx_i, idx_j = torch.meshgrid(
            torch.arange(num, device=device),
            torch.arange(num, device=device),
            indexing='ij'
        )
        idx_i = idx_i.reshape(-1)  # [num_expert]
        idx_j = idx_j.reshape(-1)  # [num_expert]

        # 向量化取出所有组合
        lr1_A_all = lr1_A[idx_i]  # [num_expert, dim_feedforward, low_rank]
        lr1_B_all = lr1_B[idx_j]  # [num_expert, low_rank, hidden_dim]
        lr2_A_all = lr2_A[idx_i]  # [num_expert, hidden_dim, low_rank]
        lr2_B_all = lr2_B[idx_j]  # [num_expert, low_rank, dim_feedforward]

        # 向量化bmm
        delta_A1 = torch.bmm(lr1_A_all, midmat_1)      # [num_expert, dim_feedforward, low_rank]
        delta_B1 = torch.bmm(delta_A1, lr1_B_all)      # [num_expert, dim_feedforward, hidden_dim]
        delta_A2 = torch.bmm(lr2_A_all, midmat_2)      # [num_expert, hidden_dim, low_rank]
        delta_B2 = torch.bmm(delta_A2, lr2_B_all)      # [num_expert, hidden_dim, dim_feedforward]

        return delta_B1, delta_B2

    # # ...existing code...
    def forward(self, x):
        a, b, c = x.shape
        x_flat = x.reshape(-1, c).contiguous()  # tokens x c

        n = self.num_expert
        p = self.p
        d = c
        m = x_flat.size(0) # tokens

        # Phi = self.Phi[:, :, :p]  # [d, n, p]
        # print(x_flat.size())  # torch.Size([100864, 384])
        # print(self.Phi.size()) # torch.Size([384, 36, 16])

        logits = torch.einsum('md,dnp->mnp', x_flat, self.Phi) # [m, n, p]
        D = F.softmax(logits, dim=0)  # [m, n, p]
        C = F.softmax(logits.reshape(m, n * p), dim=1).reshape(m, n, p)  # [m, n, p]
        Xs = torch.einsum('md,mnp->npd', x_flat, D) # Xs: [n, p, d]

        device = x.device
        experts_1, experts_2 = self.experts_generate(device)
        W1_base = self.base_expert.linear1.weight.to(device)  # [dim_feedforward, hidden_dim]
        b1_base = self.base_expert.linear1.bias.to(device) if self.base_expert.linear1.bias is not None else None
        W2_base = self.base_expert.linear2.weight.to(device)  # [dim_feedforward, hidden_dim]
        b2_base = self.base_expert.linear2.bias.to(device) if self.base_expert.linear2.bias is not None else None
        lr1_bias = self.lr1_bias.to(device)
        lr2_bias = self.lr2_bias.to(device)

        Ys = torch.zeros_like(Xs, device=x.device)  # [n, p, d]
        for i in range(n):
            W1_comb = experts_1[i] + W1_base
            b1_comb = lr1_bias[i] + b1_base

            t = F.linear(Xs[i], W1_comb, b1_comb)
            t = self.activations[i](t)

            W2_comb = experts_2[i] + W2_base
            b2_comb = lr2_bias[i] + b2_base

            out_i = F.linear(t, W2_comb, b2_comb)
            Ys[i] = out_i

        Y = torch.einsum('npd,mnp->md', Ys, C) # Y: [m, d]
        return Y.reshape(a, b, c)
