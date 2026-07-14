from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MoFEConfig


@dataclass
class RoutingState:
    balance_loss: Tensor
    z_loss: Tensor
    assignment_counts: Tensor
    router_entropy: Tensor
    topk_indices: Tensor
    shared_output_norm: Tensor | None = None
    private_output_norm: Tensor | None = None
    private_to_shared_norm: Tensor | None = None


class MoFEGPT2MLP(nn.Module):
    """GPT-2 MLP with one dense shared expert and factorized private experts."""

    def __init__(self, dense_mlp: nn.Module, config: MoFEConfig):
        super().__init__()
        self.config = config
        self.shared_expert = copy.deepcopy(dense_mlp)

        hidden_size, intermediate_size = dense_mlp.c_fc.weight.shape
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_private_experts = config.num_private_experts
        self.top_k = config.top_k
        self.num_factor_groups = math.isqrt(self.num_private_experts)
        self.rank = config.resolve_rank(hidden_size)
        config.validate(hidden_size)

        reference = dense_mlp.c_fc.weight
        factory_kwargs = {"device": reference.device, "dtype": reference.dtype}

        self.a1 = nn.Parameter(
            torch.empty(
                self.num_factor_groups,
                intermediate_size,
                self.rank,
                **factory_kwargs,
            )
        )
        self.b1 = nn.Parameter(
            torch.empty(
                self.num_factor_groups,
                self.rank,
                hidden_size,
                **factory_kwargs,
            )
        )
        self.core1 = nn.Parameter(
            torch.empty(
                self.num_private_experts,
                self.rank,
                self.rank,
                **factory_kwargs,
            )
        )
        self.a2 = nn.Parameter(
            torch.empty(
                self.num_factor_groups,
                hidden_size,
                self.rank,
                **factory_kwargs,
            )
        )
        self.b2 = nn.Parameter(
            torch.empty(
                self.num_factor_groups,
                self.rank,
                intermediate_size,
                **factory_kwargs,
            )
        )
        self.core2 = nn.Parameter(
            torch.empty(
                self.num_private_experts,
                self.rank,
                self.rank,
                **factory_kwargs,
            )
        )
        self.private_bias1 = nn.Parameter(
            torch.zeros(self.num_private_experts, intermediate_size, **factory_kwargs)
        )
        self.private_bias2 = nn.Parameter(
            torch.zeros(self.num_private_experts, hidden_size, **factory_kwargs)
        )

        self.router = nn.Linear(hidden_size, self.num_private_experts, bias=True)
        self.router.to(**factory_kwargs)
        self.private_dropout = nn.Dropout(dense_mlp.dropout.p)
        self.register_buffer(
            "private_output_scale",
            torch.tensor(float(config.private_output_scale), **factory_kwargs),
        )
        self.routing_state: RoutingState | None = None

        self.reset_private_parameters(dense_mlp)
        self.set_trainable_components()

    @torch.no_grad()
    def reset_private_parameters(self, dense_mlp: nn.Module) -> None:
        # Hugging Face Conv1D stores weights as [in_features, out_features].
        dense_w1 = dense_mlp.c_fc.weight.T
        dense_w2 = dense_mlp.c_proj.weight.T
        for group in range(self.num_factor_groups):
            self.a1[group].copy_(dense_w1[:, : self.rank])
            self.b1[group].copy_(dense_w1[: self.rank, :])
            self.a2[group].copy_(dense_w2[:, : self.rank])
            self.b2[group].copy_(dense_w2[: self.rank, :])

        nn.init.normal_(self.core1, mean=0.0, std=self.config.core_init_std)
        if self.config.zero_init_output_core:
            nn.init.zeros_(self.core2)
        else:
            nn.init.normal_(self.core2, mean=0.0, std=self.config.core_init_std)
        nn.init.zeros_(self.private_bias1)
        nn.init.zeros_(self.private_bias2)
        nn.init.normal_(self.router.weight, mean=0.0, std=self.config.router_init_std)
        nn.init.zeros_(self.router.bias)

    def set_trainable_components(self) -> None:
        for parameter in self.shared_expert.parameters():
            parameter.requires_grad = self.config.train_shared_expert
        for parameter in (self.a1, self.b1, self.a2, self.b2):
            parameter.requires_grad = self.config.train_factors
        for parameter in (self.core1, self.core2):
            parameter.requires_grad = self.config.train_cores
        for parameter in (self.private_bias1, self.private_bias2):
            parameter.requires_grad = self.config.train_private_bias
        for parameter in self.router.parameters():
            parameter.requires_grad = self.config.train_router

    @torch.no_grad()
    def set_private_scale(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("private scale must be non-negative")
        self.private_output_scale.fill_(value)

    def factor_indices(self, expert_index: int) -> tuple[int, int]:
        if not 0 <= expert_index < self.num_private_experts:
            raise IndexError(expert_index)
        return divmod(expert_index, self.num_factor_groups)

    def materialize_expert_weights(self, expert_index: int) -> tuple[Tensor, Tensor]:
        i, j = self.factor_indices(expert_index)
        weight1 = self.a1[i] @ self.core1[expert_index] @ self.b1[j]
        weight2 = self.a2[i] @ self.core2[expert_index] @ self.b2[j]
        return weight1, weight2

    def private_expert_forward(self, x: Tensor, expert_index: int) -> Tensor:
        i, j = self.factor_indices(expert_index)
        hidden = F.linear(x, self.b1[j])
        hidden = F.linear(hidden, self.core1[expert_index])
        hidden = F.linear(hidden, self.a1[i], self.private_bias1[expert_index])
        hidden = self.shared_expert.act(hidden)
        output = F.linear(hidden, self.b2[j])
        output = F.linear(output, self.core2[expert_index])
        output = F.linear(output, self.a2[i], self.private_bias2[expert_index])
        return self.private_dropout(output)

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.router(x)
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        full_probabilities = F.softmax(logits, dim=-1)
        probability_fraction = full_probabilities.mean(dim=0)
        assignment_counts = torch.bincount(
            topk_indices.reshape(-1), minlength=self.num_private_experts
        )
        assignment_fraction = assignment_counts.to(logits.dtype) / (
            x.size(0) * self.top_k
        )
        balance_loss = self.num_private_experts * torch.sum(
            probability_fraction * assignment_fraction
        )
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
        entropy = -torch.sum(
            full_probabilities * torch.log(full_probabilities.clamp_min(1e-9)), dim=-1
        ).mean()

        self.routing_state = RoutingState(
            balance_loss=balance_loss,
            z_loss=z_loss,
            assignment_counts=assignment_counts.detach(),
            router_entropy=entropy.detach(),
            topk_indices=topk_indices.detach(),
        )
        return topk_weights, topk_indices

    def forward(self, hidden_states: Tensor) -> Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        shared_output = self.shared_expert(hidden_states)
        topk_weights, topk_indices = self._route(flat_states)

        private_output = torch.zeros_like(flat_states)
        for expert_index in range(self.num_private_experts):
            positions = (topk_indices == expert_index).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_indices = positions[:, 0]
            route_slots = positions[:, 1]
            selected_states = flat_states.index_select(0, token_indices)
            expert_output = self.private_expert_forward(selected_states, expert_index)
            weights = topk_weights[token_indices, route_slots].unsqueeze(-1)
            private_output = private_output.index_add(
                0, token_indices, expert_output * weights
            )

        private_output = private_output.reshape(original_shape)
        if self.routing_state is not None:
            shared_norm = shared_output.detach().float().reshape(-1, self.hidden_size).norm(
                dim=-1
            ).mean()
            private_norm = private_output.detach().float().reshape(
                -1, self.hidden_size
            ).norm(dim=-1).mean()
            self.routing_state.shared_output_norm = shared_norm
            self.routing_state.private_output_norm = private_norm
            self.routing_state.private_to_shared_norm = private_norm / shared_norm.clamp_min(
                1e-12
            )
        return shared_output + self.private_output_scale * private_output
