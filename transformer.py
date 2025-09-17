import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Convention for tensor dimensions.
# B - Batch
# T - Context length
# H - Num of heads used for attention
# C - Embedding / feature dimension


class RMSNorm(nn.Module):
    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.epsilon = 1e-8

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)

        Returns:
            Output with shape (B, T, C)
        """

        rms = input.pow(2).mean(dim=-1, keepdim=True).sqrt()  # (B, T, 1)
        norm_input = input / (rms + self.epsilon)
        scaled_input = norm_input * self.scale
        return scaled_input


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        assert feature_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_size = feature_dim // self.num_heads
        self.normalization_factor = math.sqrt(self.head_size)

        self.attention_proj = nn.Linear(
            feature_dim,
            feature_dim * 3,
            bias=True,
        )

        self.final_proj = nn.Linear(
            feature_dim,
            feature_dim,
            bias=True,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)

        Returns:
            Output with shape (B, T, C)
        """
        B, T, C = input.shape

        # Apply projection to get q,k,v.
        qkv = self.attention_proj(input)  # (B, T, 3*C)
        qkv = qkv.view(B, T, self.num_heads, -1)  # (B, T, H, 3*C/H)
        q: torch.Tensor = qkv[..., : self.head_size]  # (B, T, H, C/H)
        k: torch.Tensor = qkv[..., self.head_size : 2 * self.head_size]
        v: torch.Tensor = qkv[..., 2 * self.head_size :]

        # TODO: Add causal attention mask
        # TODO: Support rotary embedding
        # TODO: Support group query

        # Apply multi head attention.
        q = q.permute(0, 2, 1, 3)  # (B, H, T, C/H)
        k_t = k.permute(0, 2, 3, 1)  # (B, H, C/H, T)
        att = q @ k_t / self.normalization_factor  # (B, H, T, T)
        att_prob = F.softmax(att, dim=-1)  # (B, H, T, T)
        v = v.permute(0, 2, 1, 3)  # (B, H, T, C/H)
        att_v: torch.Tensor = att_prob @ v  # (B, H, T, C/H)

        # Apply final concat and projection.
        att_v = att_v.permute(0, 2, 1, 3)  # (B, T, H, C/H)
        att_v = att_v.flatten(start_dim=2, end_dim=3)  # (B, T, C)
        output = self.final_proj(att_v)

        return output


class SwiGLU(nn.Module):
    def __init__(self, feature_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(feature_dim, 2 * out_dim, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.linear(input)
        swi_out = out[..., : self.out_dim]
        gate_out = out[..., self.out_dim :]
        return swi_out * F.sigmoid(swi_out) * gate_out


class FeedForward(nn.Module):

    def __init__(self, feature_dim: int) -> None:
        super().__init__()

        self.ff_1 = SwiGLU(feature_dim, 4 * feature_dim)
        self.ff_2 = nn.Linear(4 * feature_dim, feature_dim, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)

        Returns:
            Output with shape (B, T, C)
        """
        out = self.ff_1(input)
        out = F.relu(out)
        out = self.ff_2(out)

        return out


class Transformer(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(feature_dim, num_heads)
        self.feed_forward = FeedForward(feature_dim)
        self.att_norm = RMSNorm(feature_dim)
        self.ff_norm = RMSNorm(feature_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)

        Returns:
            Output with shape (B, T, C)
        """

        att_out = self.attention(self.att_norm(input))
        att_out += input

        ff_out = self.feed_forward(self.ff_norm(att_out))
        ff_out += att_out

        return ff_out
