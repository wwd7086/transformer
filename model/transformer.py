from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import model.pos_emb as pos_emb

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

    def forward(self, input: torch.Tensor, *args) -> torch.Tensor:
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


class AdaRMSNorm(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
    ):
        super().__init__()
        self.epsilon = 1e-8

        self.cond_proj = nn.Linear(cond_dim, feature_dim, bias=False)
        self.cond_proj.weight.data.zero_()

    def forward(
        self,
        input: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)
            condition: with shape (B, Cc)

        Returns:
            Output with shape (B, T, C)
        """

        # Normalize input.
        rms = input.pow(2).mean(dim=-1, keepdim=True).sqrt()  # (B, T, 1)
        norm_input = input / (rms + self.epsilon)
        # Apply conditioning.
        condition = F.silu(condition)
        cond_scale = self.cond_proj(condition).unsqueeze(1)  # (B, 1, C)
        scaled_input = norm_input * (1.0 + cond_scale)
        return scaled_input


class AdaScale(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
    ):
        super().__init__()

        self.cond_proj = nn.Linear(cond_dim, feature_dim, bias=False)
        self.cond_proj.weight.data.zero_()

    def forward(
        self,
        input: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)
            condition: with shape (B, Cc)

        Returns:
            Output with shape (B, T, C)
        """

        # Apply conditioning.
        condition = F.silu(condition)
        cond_scale = self.cond_proj(condition).unsqueeze(1)  # (B, 1, C)
        return input * cond_scale


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
        query_group_size: int,
        context_length: int,
        enable_causal_mask: bool,
        max_pos_emb_period: float,
    ) -> None:
        """
        Args:
            context_length: The max context length.
            num_heads: Number of query heads.
            query_group_size: How many query maps to a single kv pair.
            context_length: The max context length.
            enable_causal_mask: Whether to apply causal mask.
            max_pos_emb_period: Maximum period for rotary position embedding.
        """
        super().__init__()

        assert feature_dim % num_heads == 0
        assert num_heads % query_group_size == 0

        self.feature_dim = feature_dim
        self.query_group_size = query_group_size
        self.context_length = context_length
        self.enable_causal_mask = enable_causal_mask

        self.num_q_heads = num_heads
        self.num_kv_heads = num_heads // query_group_size
        self.head_size = feature_dim // self.num_q_heads

        self.normalization_factor = 1.0 / math.sqrt(self.head_size)

        # Attention mask.
        causal_mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

        # Rotary embedding.
        rot_sin, rot_cos = pos_emb.gen_inter_sin_cos_emb(
            self.head_size // 2,
            context_length,
            max_pos_emb_period,
        )
        self.register_buffer("rot_sin", rot_sin)
        self.register_buffer("rot_cos", rot_cos)

        # KV caches.
        self.kv_cache = KVCache(self.context_length)

        # MLP layers.
        # Note: When query_group_size is not equal to 1, multiple querys maps
        # to same kv, effective reduce the numerber of kv output size.
        output_dim = feature_dim + 2 * (feature_dim // query_group_size)
        self.attention_proj = nn.Linear(
            feature_dim,
            output_dim,
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
        qkv: torch.Tensor = self.attention_proj(input)  # (B, T, C + C/g + C/g)

        q = qkv[..., : self.feature_dim]
        q = q.reshape(
            B, T, self.query_group_size, self.num_kv_heads, self.head_size
        )  # (B, T, G, Hkv, Dh)
        q = q.transpose(1, 3)  # (B, Hkv, G, T, Dh)

        kv = qkv[..., self.feature_dim :]
        kv = kv.reshape(
            B, T, 2, self.num_kv_heads, self.head_size
        )  # (B, T, 2, Hkv, Dh)
        kv = kv.transpose(1, 3)  # (B, Hkv, 2, T, Dh)
        k, v = kv.unbind(dim=2)  # (B, Hkv, T, Dh)

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
        k_t = k.transpose(2, 3)  # (B, Hkv, Dh, Tk)
        att = q @ k_t[:, :, None, ...] * self.normalization_factor  # (B, Hkv, G, T, Tk)
        # Apply causal mask
        if self.enable_causal_mask:
            att = att.masked_fill(
                self.causal_mask[Tc : Tc + T, :Tk],
                float("-inf"),
            )
        att_prob = F.softmax(att, dim=-1)  # (B, Hkv, G, T, Tk)
        att_v: torch.Tensor = att_prob @ v[:, :, None, ...]  # (B, Hkv, G, T, Dh)

        # Apply final concat and projection.
        att_v = att_v.transpose(1, 3)  # (B, T, G, Hkv, Dh)
        att_v = att_v.flatten(start_dim=2, end_dim=4)  # (B, T, C)
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
        query_group_size: int,
        context_length: int,
        use_ada_rmsnorm: bool = False,
        enable_causal_mask: bool = True,
        max_pos_emb_period: float = 10000,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            feature_dim,
            num_heads,
            query_group_size,
            context_length,
            enable_causal_mask,
            max_pos_emb_period,
        )
        self.feed_forward = FeedForward(feature_dim)

        self.att_scale: Optional[AdaScale] = None
        self.ff_scale: Optional[AdaScale] = None
        if use_ada_rmsnorm:
            self.att_norm = AdaRMSNorm(feature_dim, feature_dim)
            self.ff_norm = AdaRMSNorm(feature_dim, feature_dim)
            self.att_scale = AdaScale(feature_dim, feature_dim)
            self.ff_scale = AdaScale(feature_dim, feature_dim)
        else:
            self.att_norm = RMSNorm(feature_dim)
            self.ff_norm = RMSNorm(feature_dim)

    def forward(
        self,
        input: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """Run forward

        Args:
            input: with shape (B, T, C)
            condition: with shape (B, Cc) or None

        Returns:
            Output with shape (B, T, C)
        """

        att_out = self.attention(
            self.att_norm(input, condition),
            use_kv_cache,
        )
        if self.att_scale is not None:
            att_out = self.att_scale(att_out, condition)
        att_out += input

        ff_out = self.feed_forward(self.ff_norm(att_out, condition))
        if self.ff_scale is not None:
            ff_out = self.ff_scale(ff_out, condition)
        ff_out += att_out

        return ff_out

    def clear_cache(self) -> None:
        self.attention.clear_cache()


if __name__ == "__main__":
    B, T, C = 3, 8, 16

    transformer = Transformer(
        feature_dim=C,
        num_heads=8,
        query_group_size=2,
        context_length=2 * T,
    )

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
