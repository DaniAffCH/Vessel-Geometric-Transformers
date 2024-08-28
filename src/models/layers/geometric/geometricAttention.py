from typing import Optional

import torch
from torch import nn

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

        query = self.q_proj(
            query
        )  # Shape: (batch_size, seq_length, embed_dim)
        key = self.k_proj(key)  # Shape: (batch_size, seq_length, embed_dim)
        value = self.v_proj(
            value
        )  # Shape: (batch_size, seq_length, embed_dim)
