from typing import List, Tuple

import torch
import typer
from torch import nn

__all__: List[str] = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

"""
    deep-learning-at-scale chapter_11 moe run
"""

class Experts(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_experts=6,
    ):
        super().__init__()

        hidden_dim = dim * 4

        self.layer1 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim))
        self.activation = nn.LeakyReLU()
        self.layer2 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim))

        self.apply(self.init_weights)  # type: ignore[arg-type]

    @torch.no_grad()
    def init_weights(p: nn.Module, initializer_range: float = 0.02):
        if isinstance(p, nn.Parameter):
            p.data = nn.init.trunc_normal_(
                p.data.to(torch.float32),
                mean=0.0,
                std=initializer_range,
            ).to(p.dtype)

    def forward(self, x):
        hidden = torch.einsum("...nd,...dh->...nh", x, self.layer1)
        hidden = self.activation(hidden)
        out = torch.einsum("...nh,...hd->...nd", hidden, self.layer2)
        return out


class Top2ThresholdGating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps=1e-9,
        threshold=0.2,
        capacity_factor_tuple=(1.25, 2.0),
    ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self._minimum_expert = 4
        self.threshold = threshold
        self.capacity_factor_tuple = capacity_factor_tuple

        self.gating_weights = nn.Parameter(torch.randn(dim, num_gates))

    def _top_k_value_index_tuple(
        self, t: torch.Tensor, k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values, index = t.topk(k=k, dim=-1)
        return values.squeeze(dim=-1), index.squeeze(dim=-1)

    def _sum_along_last_second_dim(self, t: torch.Tensor) -> torch.Tensor:
        pre_padding = (0, 0)
        pre_slice = (slice(None),)
        padded_t = torch.nn.functional.pad(t, (*pre_padding, 1, 0)).cumsum(dim=-2)
        return padded_t[(..., slice(None, -1), *pre_slice)]

    def _safe_one_hot(self, indexes, max_length):
        max_index = indexes.max() + 1
        return torch.nn.functional.one_hot(indexes, max(max_index + 1, max_length))[
            ..., :max_length
        ]

    def forward(self, x):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        idx = 0 if self.training else 1
        capacity_factor = self.capacity_factor_tuple[idx]

        raw_gates = torch.einsum(
            "...bnd,...de->...bne", x, self.gating_weights
        )  # [...,batch, dim, expert]
        raw_gates = raw_gates.softmax(dim=-1)

        # find top-1 expert
        gate_1, index_1 = self._top_k_value_index_tuple(raw_gates)  # [batch, dim]
        mask_1 = torch.nn.functional.one_hot(
            index_1, num_gates
        ).float()  # [batch, dim, expert]

        # find second top expert
        gates_without_top_1 = raw_gates * (1.0 - mask_1)
        gate_2, index_2 = self._top_k_value_index_tuple(gates_without_top_1)
        mask_2 = torch.nn.functional.one_hot(index_2, num_gates).float()

        # normalize the gate scores
        gate_1 = gate_1 / (gate_1 + gate_2 + self.eps)
        gate_2 = gate_2 / (gate_1 + gate_2 + self.eps)

        # Implement a threshold based gating
        mask_2 *= (gate_2 > self.threshold).unsqueeze(dim=-1).float()

        # estimate capacity of experts given group size, capacity factor and minimum
        # expert required for decisions
        expert_capacity = max(
            min(group_size, int((group_size * capacity_factor) / num_gates)),
            self._minimum_expert,
        )

        # Calculate expert assigement tensor to distribute the samples to experts.
        # chose most suitable sample and limit by capacity
        position_in_expert_1 = self._sum_along_last_second_dim(mask_1) * mask_1
        mask_1 *= (position_in_expert_1 < float(expert_capacity)).float()
        mask_1_flat = mask_1.sum(dim=-1)
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        gate_1 *= mask_1_flat

        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        position_in_expert_2 = (
            self._sum_along_last_second_dim(mask_2) + mask_1_count
        ) * mask_2
        mask_2 *= (position_in_expert_2 < float(expert_capacity)).float()
        mask_2_flat = mask_2.sum(dim=-1)
        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat

        # combine the two experts into [batch, group, experts, expert_capacity]
        routing_tensor_1 = (
            (gate_1 * mask_1_flat)[..., None, None]
            * torch.nn.functional.one_hot(index_1, num_gates)[..., None]
            * self._safe_one_hot(position_in_expert_1.long(), expert_capacity)[
                ..., None, :
            ]
        )
        routing_tensor_2 = (
            (gate_2 * mask_2_flat)[..., None, None]
            * torch.nn.functional.one_hot(index_2, num_gates)[..., None]
            * self._safe_one_hot(position_in_expert_2.long(), expert_capacity)[
                ..., None, :
            ]
        )

        return routing_tensor_1 + routing_tensor_2


class MoE(nn.Module):
    def __init__(
        self,
        dim,
        num_experts=6,
    ):
        super().__init__()

        self.num_experts = num_experts

        self.gate = Top2ThresholdGating(
            dim=dim,
            num_gates=num_experts,
        )
        self.experts = Experts(
            dim=dim,
            num_experts=num_experts,
        )

    def forward(self, inputs, **kwargs):
        routing_tensor = self.gate(inputs)
        expert_inputs = torch.einsum(
            "bnd,bnec->ebcd", inputs, routing_tensor.bool().to(inputs)
        )

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(self.num_experts, -1, inputs.shape[-1])
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum("ebcd,bnec->bnd", expert_outputs, routing_tensor)
        return output


@app.command()
def run(
    batch_size: int = typer.Option(3, help="Batch size per trainer"),
    n: int = typer.Option(5, help="number of hidden dim"),
    dim: int = typer.Option(10, help="dim"),
):
    moe = MoE(dim=dim)
    inputs = torch.randn(batch_size, n, dim)
    out = moe(inputs)
    print(out.shape)
