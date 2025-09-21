import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pos_emb

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
        context_length: int,
    ) -> None:
        super().__init__()

        assert feature_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_size = feature_dim // self.num_heads
        self.normalization_factor = 1.0 / math.sqrt(self.head_size)

        # Attention mask.
        causal_mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

        # Rotary embedding.
        rot_thetas = pos_emb.gen_thetas(self.head_size // 2)
        rot_sin = pos_emb.gen_sin_emb(rot_thetas, context_length)
        rot_cos = pos_emb.gen_cos_emb(rot_thetas, context_length)
        self.register_buffer("rot_sin", rot_sin)
        self.register_buffer("rot_cos", rot_cos)

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
        qkv: torch.Tensor = self.attention_proj(input)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)  # (B, T, 3, H, Dh)
        qkv = qkv.transpose(1, 3)  # (B, H, 3, T, Dh)
        q, k, v = qkv.unbind(dim=2)  # (B, H, T, Dh)

        # Apply rotary embedding to q, k.
        q = pos_emb.apply_rot_emb(q, self.rot_sin, self.rot_cos)
        k = pos_emb.apply_rot_emb(k, self.rot_sin, self.rot_cos)

        # Apply multi head attention.
        k_t = k.transpose(2, 3)  # (B, H, Dh, T)
        att = q @ k_t * self.normalization_factor  # (B, H, T, T)
        # Apply causal mask
        att = att.masked_fill(
            self.causal_mask[:T, :T],
            float("-inf"),
        )
        att_prob = F.softmax(att, dim=-1)  # (B, H, T, T)
        att_v: torch.Tensor = att_prob @ v  # (B, H, T, Dh)

        # Apply final concat and projection.
        att_v = att_v.transpose(1, 2)  # (B, T, H, Dh)
        att_v = att_v.flatten(start_dim=2, end_dim=3)  # (B, T, C)
        output = self.final_proj(att_v)

        return output


class SwiGLU(nn.Module):
    def __init__(self, feature_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(feature_dim, 2 * out_dim, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        swi_out, gate_out = self.linear(input).split(self.out_dim, dim=-1)
        return swi_out * torch.sigmoid(swi_out) * gate_out


class FeedForward(nn.Module):

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden_dim = int(8 * feature_dim / 3)
        self.ff_1 = SwiGLU(feature_dim, hidden_dim)
        self.ff_2 = nn.Linear(hidden_dim, feature_dim, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)

        Returns:
            Output with shape (B, T, C)
        """
        out = self.ff_1(input)
        out = self.ff_2(out)

        return out


class Transformer(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        context_length: int,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            feature_dim,
            num_heads,
            context_length,
        )
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


# TODO: Support group query
# TODO: Try out flash attention

# Next step:
# 1. Create the full nano GPT
# 2. Train on the tiny sharkspear dataset
# 3. Run inference with KV cache
# 5. list common set of mistakes and bugs
#  - transpose, view, contigous
# 6. Right some proper tests that could prob the implementations

if __name__ == "__main__":
    B, T, C = 2, 8, 16

    transformer = Transformer(feature_dim=C, num_heads=2, context_length=T)

    test_input = torch.zeros(B, T, C)
    test_output = transformer(test_input)

    assert test_output.shape == test_input.shape
    print(test_output.shape)
