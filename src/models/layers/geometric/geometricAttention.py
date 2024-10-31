import torch
import torch.nn.functional as F
from torch import nn

from src.lib.geometricAlgebraElements import GeometricAlgebraBase
from src.lib.geometricAlgebraOperations import InnerProduct
from src.models.layers.geometric.equiLinear import EquiLinearLayer


class GeometricAttentionLayer(nn.Module):  # type:ignore[misc]
    """
    Computes the attention mechanism in the Geometric Algebra space. The
    doto product is replaced with the invariant inner product.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super(GeometricAttentionLayer, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.ga_size = GeometricAlgebraBase.GA_size

        self.q_proj = EquiLinearLayer(embed_dim, embed_dim)
        self.k_proj = EquiLinearLayer(embed_dim, embed_dim)
        self.v_proj = EquiLinearLayer(embed_dim, embed_dim)

        self.out_proj = EquiLinearLayer(embed_dim, embed_dim)

        self.ipIdx = InnerProduct.getBasisIndices()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
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

        # Project q and k to non e0 components for geom inner product
        query = (
            query[..., self.ipIdx]
            .reshape(
                batch_size, -1, self.num_heads, self.head_dim * len(self.ipIdx)
            )
            .transpose(1, 2)
        )  # Shape: (batch_size, num_heads, num_f, head_dim)
        key = (
            key[..., self.ipIdx]
            .reshape(
                batch_size, -1, self.num_heads, self.head_dim * len(self.ipIdx)
            )
            .transpose(1, 2)
        )  # Shape: (batch_size, num_heads, num_f, head_dim)
        value = value.reshape(
            batch_size, -1, self.num_heads, self.head_dim * self.ga_size
        ).transpose(
            1, 2
        )  # Shape: (batch_size, num_heads, num_f, head_dim * ga_size)

        scores = (
            torch.matmul(query, key.transpose(-2, -1))
            / (self.head_dim * 8) ** 0.5
        )  # Shape: (batch_size, num_heads, num_f, num_f)

        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(
            attn_weights, value
        )  # Shape: (batch_size, num_heads, num_f, head_dim * ga_size)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, -1, self.embed_dim, self.ga_size)
        )  # Shape: (batch_size, num_f, embed_dim * ga_size)

        # Apply output projection
        output = self.out_proj(
            attn_output
        )  # Shape: (batch_size, num_f, embed_dim, ga_size)

        return output
