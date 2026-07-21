import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack

from .distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    has_only_one_value
)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = - 1)

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# norm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma

# expert

def FeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

class _Expert(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation = 'GELU',
    ):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

    def forward(self, x):
        tgt = self.linear1(x)
        tgt = self.dropout(self.activation(tgt))

        output = self.linear2(tgt)

        return output

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def GLUFeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

# experts

class Experts(nn.Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        offload_unused_experts_to_cpu = True
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # whether to offload unused experts to cpu, will require optimizer handles conversion of gradients to right device when accumulating
        self.offload_unused_experts_to_cpu = offload_unused_experts_to_cpu

        self.all_gather = AllGather()
        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def init_all_expert_from_checkpoint(self, state_dict, prefix: str):
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
                if tuple(t_l1_w.shape) != tuple(self.experts[0].linear1.weight.shape):
                    raise RuntimeError(f"shape mismatch for linear1.weight: ckpt {tuple(t_l1_w.shape)} vs expert {tuple(self.experts[0].linear1.weight.shape)}")
                for expert in self.experts:
                    expert.linear1.weight.data.copy_(t_l1_w.to(expert.linear1.weight.device))
            if t_l1_b is not None:
                if tuple(t_l1_b.shape) != tuple(self.experts[0].linear1.bias.shape):
                    raise RuntimeError(f"shape mismatch for linear1.bias: ckpt {tuple(t_l1_b.shape)} vs expert {tuple(self.experts[0].linear1.bias.shape)}")
                for expert in self.experts:
                    expert.linear1.bias.data.copy_(t_l1_b.to(expert.linear1.bias.device))

            # copy linear2
            if t_l2_w is not None:
                if tuple(t_l2_w.shape) != tuple(self.experts[0].linear2.weight.shape):
                    raise RuntimeError(f"shape mismatch for linear2.weight: ckpt {tuple(t_l2_w.shape)} vs expert {tuple(self.experts[0].linear2.weight.shape)}")
                for expert in self.experts:
                    expert.linear2.weight.data.copy_(t_l2_w.to(expert.linear2.weight.device))
            if t_l2_b is not None:
                if tuple(t_l2_b.shape) != tuple(self.experts[0].linear2.bias.shape):
                    raise RuntimeError(f"shape mismatch for linear2.bias: ckpt {tuple(t_l2_b.shape)} vs expert {tuple(self.experts[0].linear2.bias.shape)}")
                for expert in self.experts:
                    expert.linear2.bias.data.copy_(t_l2_b.to(expert.linear2.bias.device))
            print(f"init_experts_from_checkpoint: copied params to the base experts from prefix '{prefix}'")

    def all_experts_cosine_similarity(
            self,
            which: str = "both",
            include_bias: bool = False,
            device: str = None,
            eps: float = 1e-8,
        ) -> torch.Tensor:

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

    def all_experts_to_cpu_besides(self, selection):
        if not self.offload_unused_experts_to_cpu:
            return

        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts

        # for now naively all gather across batch dimension if distributed, optimize later

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            assert has_only_one_value(seq_sizes), 'number of tokens per expert must be the same'

            x, batch_sizes = self.all_gather(x)
            total_batch_size = x.shape[0]

            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # the experts in use on the rank

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim = -1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)
        else:
            num_experts_per_rank = num_experts
            expert_slice = slice(0, num_experts)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')
            x = x.split(num_experts_batches_across_ranks, dim = 0)
            x = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []
        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x).requires_grad_()

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs = split_by_rank(outs)

        assert outs.shape == shape
        return outs

# main class

class Soft_MoE(Module):
    def __init__(
        self,
        hidden_dim,
        dim_feedforward,
        dropout,
        activation,
        moe_num_expert,
        moe_top_k,
        expert_choice,
        seq_len = None,
        num_slots = 1,
        geglu = False,
        is_distributed = None,
        offload_unused_experts_to_cpu = False,
        use_layernorm = False
    ):
        super().__init__()
        assert exists(seq_len) ^ exists(num_slots), 'either seq_len, or num_slots must be passed into SoftMoE'

        if exists(seq_len):
            num_slots = default(num_slots, seq_len // moe_num_expert)
        elif exists(num_slots):
            seq_len = num_slots * moe_num_expert

        norm_klass = LayerNorm if use_layernorm else RMSNorm
        self.norm = norm_klass(hidden_dim)

        self.slot_norm = norm_klass(hidden_dim)
        self.slot_embeds = nn.Parameter(torch.randn(moe_num_expert, num_slots, hidden_dim))

        # expert_klass = GLUFeedForward if geglu else FeedForward
        expert_klass = _Expert
        expert_mult = int(dim_feedforward // hidden_dim)

        self.experts = Experts(
            experts = [expert_klass(hidden_dim = hidden_dim, dim_feedforward = dim_feedforward, dropout = dropout) for _ in range(moe_num_expert)],
            is_distributed = is_distributed,
            offload_unused_experts_to_cpu = offload_unused_experts_to_cpu
        )

    # def init_experts_from_checkpoint(self, state_dict, prefix: str):
    #     self.experts.init_all_expert_from_checkpoint(state_dict, prefix)

    def experts_cosine_similarity(
            self,
            which: str = "both",
            include_bias: bool = False,
            device: str = None,
            eps: float = 1e-8,
        ) -> torch.Tensor:
        return self.experts.all_experts_cosine_similarity(which, include_bias, device, eps)


    def forward(self, x, mask = None, add_noise = False, noise_mult = 1.):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        is_single_token = x.ndim == 2
        is_image = x.ndim == 4

        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')
        elif is_single_token:
            x = rearrange(x, 'b d -> b 1 d')

        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)

        x = self.norm(x)
        slot_embeds = self.slot_norm(self.slot_embeds)

        logits = einsum('b n d, e s d -> b n e s', x, slot_embeds)

        # noised dispatch and combine gate logits, with annealing if needed

        if add_noise:
            noise = gumbel_noise(logits) * noise_mult
            logits = logits + noise

        # account for key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

        # get dispatch and combine weights (softmax across right dimensions)

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # derive slots by weighted average of input tokens using the dispatch weights from above

        slots = einsum('b n d, b n e s -> b e s d', x, dispatch_weights)

        # route the slots per expert to each expert

        out = self.experts(slots)

        # combine back out

        out = rearrange(out, ' b e s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')
        elif is_single_token:
            out = rearrange(out, 'b 1 d -> b d')

        return out