from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.lib.geometricAlgebraElements import GeometricAlgebraBase
from src.lib.geometricAlgebraOperations import InnerProduct
from src.models.layers.geometric.equiLinear import EquiLinearLayer


class GeometricAttentionLayer(nn.Module):  # type:ignore[misc]
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super(GeometricAttentionLayer, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.ga_size = GeometricAlgebraBase.GA_size

        # Define the projection layers for query, key, and value
        self.q_proj = EquiLinearLayer(embed_dim, embed_dim)
        self.k_proj = EquiLinearLayer(embed_dim, embed_dim)
        self.v_proj = EquiLinearLayer(embed_dim, embed_dim)

        # Define the output projection layer
        self.out_proj = EquiLinearLayer(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size = query.size(0)

        query = self.q_proj(
            query
        )  # Shape: (batch_size, num_f, embed_dim, ga_size)
        key = self.k_proj(
            key
        )  # Shape: (batch_size, num_f, embed_dim, ga_size)
        value = self.v_proj(
            value
        )  # Shape: (batch_size, num_f, embed_dim, ga_size)

        # Shape: (batch_size, num_heads, num_f, embed_dim, ga_size)
        query = query.view(
            batch_size, -1, self.num_heads, self.head_dim, self.ga_size
        ).transpose(1, 2)
        key = key.view(
            batch_size, -1, self.num_heads, self.head_dim, self.ga_size
        ).transpose(1, 2)
        value = value.view(
            batch_size, -1, self.num_heads, self.head_dim, self.ga_size
        ).transpose(1, 2)

        iProd = InnerProduct.apply(query, key)
        iProd = iProd.unsqueeze(-1)

        scaling_factor = (self.head_dim * 8) ** 0.5
        scores = iProd / scaling_factor

        attentionWeights = F.softmax(scores, dim=-3)  # Along the featuers

        weighted_values = (
            attentionWeights * value
        )  # Shape: (batch_size, num_heads, num_f, head_dim, ga_size)

        weighted_values = weighted_values.transpose(1, 2).contiguous()
        weighted_values = weighted_values.view(
            batch_size, -1, self.embed_dim, self.ga_size
        )

        output = self.out_proj(
            weighted_values
        )  # Shape: (batch_size, num_f, embed_dim, ga_size)

        return output
