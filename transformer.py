from typing import Optional

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


class KVCache:
    def __init__(self, cache_size: int) -> None:
        self.cache_size = cache_size
        # KV caches.
        self.k_cache: Optional[torch.Tensor] = None  # (B, H, Tc, Dh)
        self.v_cache: Optional[torch.Tensor] = None  # (B, H, Tc, Dh)

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = k.shape[0]
        if self.k_cache is None:
            # Initialize the cache.
            self.k_cache = k
            self.v_cache = v
        else:
            # Ensure the batch dimension matches.
            assert self.v_cache is not None
            assert B == self.k_cache.shape[0]
            # Append to the cache.
            self.k_cache = torch.concat((self.k_cache, k), dim=2)
            self.k_cache = self.k_cache[..., -self.cache_size :, :]
            self.v_cache = torch.concat((self.v_cache, v), dim=2)
            self.v_cache = self.v_cache[..., -self.cache_size :, :]
        return self.k_cache, self.v_cache

    def clear_cache(self) -> None:
        """Clear the kv cache."""
        self.k_cache = None
        self.v_cache = None


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        context_length: int,
    ) -> None:
        """
        Args:
            context_length: The max context length.
        """
        super().__init__()

        assert feature_dim % num_heads == 0

        self.num_heads = num_heads
        self.context_length = context_length
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

        # KV caches.
        self.kv_cache = KVCache(self.context_length)

        # MLP layers.
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

    def forward(
        self,
        input: torch.Tensor,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)

        Returns:
            Output with shape (B, T, C)

        Note:
            T should be equal or smaller than the context length.
        """
        B, T, C = input.shape

        # Apply projection to get q,k,v.
        qkv: torch.Tensor = self.attention_proj(input)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)  # (B, T, 3, H, Dh)
        qkv = qkv.transpose(1, 3)  # (B, H, 3, T, Dh)
        q, k, v = qkv.unbind(dim=2)  # (B, H, T, Dh)

        if use_kv_cache:
            k, v = self.kv_cache.append(k, v)

        # Note:
        # - T: Is the number of new query keys
        # - Tc: Is the number of cached keys
        # - Tk: Is the total number of keys
        Tk = k.shape[2]
        assert Tk <= self.context_length
        Tc = Tk - T

        # Apply rotary embedding to q, k.
        q = pos_emb.apply_rot_emb(
            q, self.rot_sin[Tc : Tc + T, :], self.rot_cos[Tc : Tc + T, :]
        )
        k = pos_emb.apply_rot_emb(
            k,
            self.rot_sin[:Tk, :],
            self.rot_cos[:Tk, :],
        )

        # Apply multi head attention.
        k_t = k.transpose(2, 3)  # (B, H, Dh, Tk)
        att = q @ k_t * self.normalization_factor  # (B, H, T, Tk)
        # Apply causal mask
        att = att.masked_fill(
            self.causal_mask[Tc : Tc + T, :Tk],
            float("-inf"),
        )
        att_prob = F.softmax(att, dim=-1)  # (B, H, T, Tk)
        att_v: torch.Tensor = att_prob @ v  # (B, H, T, Dh)

        # Apply final concat and projection.
        att_v = att_v.transpose(1, 2)  # (B, T, H, Dh)
        att_v = att_v.flatten(start_dim=2, end_dim=3)  # (B, T, C)
        output = self.final_proj(att_v)

        return output

    def clear_cache(self) -> None:
        """Clear the kv cache."""
        self.kv_cache.clear_cache()


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

    def forward(
        self,
        input: torch.Tensor,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)

        Returns:
            Output with shape (B, T, C)
        """

        att_out = self.attention(
            self.att_norm(input),
            use_kv_cache,
        )
        att_out += input

        ff_out = self.feed_forward(self.ff_norm(att_out))
        ff_out += att_out

        return ff_out

    def clear_cache(self) -> None:
        self.attention.clear_cache()


# TODO: Support group query
# TODO: Try out flash attention

if __name__ == "__main__":
    B, T, C = 2, 8, 16

    transformer = Transformer(feature_dim=C, num_heads=2, context_length=2 * T)

    # Test simple forward.
    test_input = torch.zeros(B, T, C)
    test_output = transformer(test_input)
    print(test_output.shape)
    assert test_output.shape == test_input.shape

    # Test forward with cache.
    test_output = transformer(test_input, use_kv_cache=True)
    test_input2 = torch.zeros(B, 1, C)
    test_output2 = transformer(test_input2, use_kv_cache=True)
    print(test_output2.shape)
    assert test_output2.shape == test_input2.shape

    transformer.clear_cache()
    test_output2 = transformer(test_input2, use_kv_cache=True)
    print(test_output2.shape)
    assert test_output2.shape == test_input2.shape
