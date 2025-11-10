# Implementation of the diffusion image transformer

from dataclasses import dataclass

import torch
import torch.nn as nn

import model.transformer as transformer
from model.pos_emb import gen_thetas, gen_sin_cos_emb, gen_2d_sin_cos_emb


class ImageEncoder(nn.Module):
    def __init__(self, patch_size: int, emb_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            1,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, input_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_img: with shape (B, H, W) binary image

        Returns:
            with shape (B, T, C)
        """

        x = input_img.unsqueeze(1)  # (B, 1, H, W)
        x = self.conv(x)  # (B, C, H/P, W/P)
        x = x.flatten(2)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)

        return x


class ImageDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        emb_dim: int,
        output_dim: int,
        image_dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(emb_dim, patch_size * patch_size * output_dim)
        self.image_dim = image_dim

    def forward(self, input_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_emb: with shape (B, T, C)

        Returns:
            with shape (B, H, W)
        """

        B, T, C = input_emb.shape
        x = self.proj(input_emb)  # (B, T, P*P*D)

        # (B, H/P, W/P, P, P, D) -> (B, H, W, D)
        H = W = int(self.image_dim / self.patch_size)
        x = x.view(B, H, W, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, self.image_dim, self.image_dim, -1)  # (B, H, W, D)

        return x


class TimeEncoder(nn.Module):
    def __init__(
        self,
        max_period: int,
        num_thetas: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        thetas = gen_thetas(num_thetas, max_period)
        self.register_buffer("thetas", thetas)
        # MLP to process time embedding.
        time_emb_dim = 2 * num_thetas
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: with shape (B,)

        Returns:
            with shape (B, C)
        """
        # Convert t to float.
        t = t.float()
        # Generate sin and cos embeddings.
        time_emb = gen_sin_cos_emb(self.thetas, t)
        # Apply MLP.
        proj_emb = self.mlp(time_emb)  # (B, C)
        return proj_emb


@dataclass
class TinyDiTConfig:
    image_dim: int
    patch_size: int
    emb_dim: int
    output_dim: int
    num_layers: int
    num_heads: int
    query_group_size: int
    context_length: int
    time_max_period: int
    time_num_thetas: int
    time_output_dim: int


class TinyDiT(nn.Module):

    def __init__(self, config: TinyDiTConfig) -> None:
        super().__init__()

        self.config = config

        self.encoder = ImageEncoder(
            patch_size=self.config.patch_size,
            emb_dim=self.config.emb_dim,
        )

        self.decoder = ImageDecoder(
            patch_size=self.config.patch_size,
            emb_dim=self.config.emb_dim,
            output_dim=self.config.output_dim,
            image_dim=self.config.image_dim,
        )

        self.time_encoder = TimeEncoder(
            self.config.time_max_period,
            self.config.time_num_thetas,
            self.config.time_output_dim,
        )

        # Fixed positional embedding.
        fixed_pos_emb = gen_2d_sin_cos_emb(
            self.config.emb_dim // 4,
            self.config.image_dim // self.config.patch_size,
            self.config.image_dim // self.config.patch_size,
            max_period=100,
        )  # (T, 4N)
        self.register_buffer("pos_emb", fixed_pos_emb)  # (T, 4N)

        self.transformer_layers = nn.ModuleList(
            [
                transformer.Transformer(
                    self.config.emb_dim,
                    self.config.num_heads,
                    self.config.query_group_size,
                    self.config.context_length,
                    use_ada_rmsnorm=True,
                    enable_causal_mask=False,
                    max_pos_emb_period=100,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.final_norm = transformer.RMSNorm(self.config.emb_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: with shape (B, H, W) binary image
            t: with shape (B,) of type int64

        Returns:
            with shape (B, H, W, out)
        """

        x = self.encoder(x)  # (B, T, C)
        x = x + self.pos_emb  # (B, T, C)
        t_emb = self.time_encoder(t)  # (B, Ct)

        for layer in self.transformer_layers:
            x = layer(x, condition=t_emb)  # (B, T, C)

        x = self.final_norm(x)  # (B, T, C)
        x = self.decoder(x)  # (B, H, W, 2)

        return x

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
    # Test the image encoder.
    image_encoder = ImageEncoder(patch_size=4, emb_dim=128)
    dummy_img = torch.randn(2, 28, 28)  # (B, H, W)
    enc_out = image_encoder(dummy_img)
    print(f"Image encoder output shape: {enc_out.shape}")
    assert enc_out.shape == (2, 49, 128)

    # Test the time encoder.
    time_encoder = TimeEncoder(
        max_period=1000,
        num_thetas=32,
        output_dim=128,
    )
    dummy_t = torch.randint(0, 1000, (2,))  # (B,)
    time_out = time_encoder(dummy_t)
    print(f"Time encoder output shape: {time_out.shape}")
    assert time_out.shape == (2, 128)

    # Test the TinyDiT model.
    config = TinyDiTConfig(
        image_dim=28,
        patch_size=4,
        emb_dim=128,
        output_dim=2,
        num_layers=6,
        num_heads=4,
        query_group_size=1,
        context_length=49,
        time_max_period=1000,
        time_num_thetas=32,
        time_output_dim=128,
    )
    model = TinyDiT(config)
    print(model)

    test_out = model(dummy_img, dummy_t)
    print(f"TinyDiT output shape: {test_out.shape}")
    assert test_out.shape == (2, 28, 28, 2)
