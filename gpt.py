from dataclasses import dataclass
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer
from token_emb import TokenEmbedder


@dataclass
class TinyGPTConfig:
    emb_dim: int
    num_layers: int
    num_heads: int
    query_group_size: int
    vocab_size: int
    context_length: int


class TinyGPT(nn.Module):

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()

        self.config = config

        self.embedder = TokenEmbedder(
            self.config.vocab_size,
            self.config.emb_dim,
        )

        self.transformer_layers = nn.ModuleList(
            [
                transformer.Transformer(
                    self.config.emb_dim,
                    self.config.num_heads,
                    self.config.query_group_size,
                    self.config.context_length,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.final_norm = transformer.RMSNorm(self.config.emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: with shape [B, T], dtype int64

        Returns:
            with shape [B, T, vocab_size]
        """

        # Conver the tokens into embeddings
        x_emb = self.embedder.encode(x)

        # Apply multiple layers of transformer
        for transformer_layer in self.transformer_layers:
            x_emb = transformer_layer(x_emb, use_kv_cache)

        # Decode the ouput
        x_emb = self.final_norm(x_emb)
        x_logits = self.embedder.decode(x_emb)

        return x_logits

    @torch.no_grad()
    def sample(
        self,
        prompt: torch.Tensor,
        max_step: int,
        temperature: float = 1.0,
    ) -> Generator[torch.Tensor, None, None]:
        """Sample the output sequence autoregressivly based on the prompt.

        Args:
            prompt: with shape [B, Tp], dtype int64

        Returns:
            with shape [B, Ta], dtype int 64
        """
        self.eval()

        # Prefill the prompt.
        out_logits = self.forward(prompt, use_kv_cache=True)[:, -1, ...]
        out_logits /= temperature
        out_prob = F.softmax(out_logits, dim=-1)  # (B, V)
        last_out_idx = torch.multinomial(out_prob, num_samples=1)
        yield last_out_idx

        # Autoregressively decode.
        for _ in range(max_step):
            out_logits = self.forward(last_out_idx, use_kv_cache=True)[:, 0, ...]
            out_logits /= temperature
            out_prob = F.softmax(out_logits, dim=-1)  # (B, V)
            last_out_idx = torch.multinomial(out_prob, num_samples=1)
            yield last_out_idx

        for tm in self.transformer_layers:
            tm.clear_cache()

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer.
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
        )

        return optimizer


if __name__ == "__main__":
    config = TinyGPTConfig(
        emb_dim=8,
        num_layers=3,
        num_heads=2,
        query_group_size=1,
        vocab_size=8,
        context_length=4,
    )
    batch_size = 2

    tiny_gpt = TinyGPT(config)

    test_input = torch.ones(
        (batch_size, config.context_length),
        dtype=torch.int32,
    )

    test_output = tiny_gpt(test_input)
    print(test_output)
    assert test_output.shape == (
        batch_size,
        config.context_length,
        config.vocab_size,
    )
