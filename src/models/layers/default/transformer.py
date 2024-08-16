from typing import Optional

import torch
from torch import nn

from src.models.layers.default.highway import HighwayLayer


class MultiHeadAttentionLayer(nn.Module):  # type: ignore[misc]
    """
    Multi-head attention layer.

    Args:
        embed_dim (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.

    Attributes:
        embed_dim (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimensionality of each attention head.
        q_proj (nn.Linear): Linear layer to project the queries.
        k_proj (nn.Linear): Linear layer to project the keys.
        v_proj (nn.Linear): Linear layer to project the values.
        out_proj (nn.Linear): Linear layer for the final output projection.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super(MultiHeadAttentionLayer, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Define the projection layers for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Define the output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for the multi-head attention layer.

        Args:
            query (torch.Tensor): The query tensor of shape
                                  (batch_size, seq_length, embed_dim).
            key (torch.Tensor): The key tensor of shape
                                (batch_size, seq_length, embed_dim).
            value (torch.Tensor): The value tensor of shape
                                  (batch_size, seq_length, embed_dim).
            mask (torch.Tensor, optional): The attention mask tensor of shape
                                    (batch_size, 1, seq_length, seq_length).
                                    Default is None.

        Returns:
            torch.Tensor: The output tensor after applying
                          multi-head attention.
        """
        batch_size, seq_length = query.size(0), query.size(1)

        query = self.q_proj(
            query
        )  # Shape: (batch_size, seq_length, embed_dim)
        key = self.k_proj(key)  # Shape: (batch_size, seq_length, embed_dim)
        value = self.v_proj(
            value
        )  # Shape: (batch_size, seq_length, embed_dim)

        # Shape: (batch_size, num_heads, seq_length, head_dim)
        query = query.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # Compute scaled dot-product attention
        scores = (
            torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
        )  # Shape: (batch_size, num_heads, seq_length, seq_length)
        if mask is not None:

            # Shape: (batch_size, seq_length)
            # -> (batch_size, seq_length, seq_length)
            mask = mask.unsqueeze(1) * mask.unsqueeze(2)

            # Shape: (batch_size, seq_length, seq_length)
            # -> (batch_size, num_heads, seq_length, seq_length)
            mask = mask.unsqueeze(1).expand(
                batch_size, self.num_heads, seq_length, seq_length
            )

            # If a token is masked the model shouldn't pay attention to it
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(
            attn_weights, value
        )  # Shape: (batch_size, num_heads, seq_length, head_dim)

        # Transpose and reshape back to (batch_size, seq_length, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        return output


class TransformerEncoderLayer(nn.Module):  # type: ignore[misc]
    """
    Transformer encoder layer with highway connections.

    Args:
        embed_dim (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        feedforward_dim (int): The dimensionality of the feedforward network.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, feedforward_dim: int
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadAttentionLayer(embed_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.highway1 = HighwayLayer(embed_dim)
        self.highway2 = HighwayLayer(embed_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer encoder layer.

        Args:
            x (torch.Tensor): The input tensor of shape
                              (batch_size, seq_length, embed_dim).
            mask (torch.Tensor, optional): The attention mask tensor of shape
                                    (batch_size, seq_length).
                                    Default is None.

        Returns:
            torch.Tensor: The output tensor after applying
                          the Transformer encoder layer.
        """
        attn_output = self.attention(x, x, x, mask)
        x = self.highway1(x, self.norm1(attn_output))

        ff_output = self.feedforward(x)
        x = self.highway2(x, self.norm2(ff_output))

        return x


class TransformerEncoder(nn.Module):  # type: ignore[misc]
    """
    Transformer encoder consisting of multiple encoder layers.

    Args:
        embed_dim (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        feedforward_dim (int): The dimensionality of the feedforward network.
        num_layers (int): The number of encoder layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
    ) -> None:
        super(TransformerEncoder, self).__init__()

        self.layers = nn.Sequential(
            *[
                TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer encoder.

        Args:
            x (torch.Tensor): The input tensor of shape
                              (batch_size, seq_length, embed_dim).
            mask (torch.Tensor, optional): The attention mask tensor of shape
                                    (batch_size, 1, seq_length, seq_length).
                                    Default is None.

        Returns:
            torch.Tensor: The output tensor after applying
                          the Transformer encoder.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
