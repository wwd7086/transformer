# Forward diffusion
from typing import Generator
import torch

from model.diffusion_schedule import DiffusionSchedule


def preprocess_image(x: torch.Tensor) -> torch.Tensor:
    """
    Preprocess the image to map to -1 to 1.

    Args:
        x: with shape (B, H, W), pixel values in [0, 1]

    Returns:
        with shape (B, H, W), pixel values in [-1, 1]
    """
    return x * 2.0 - 1.0


def postprocess_image(x: torch.Tensor) -> torch.Tensor:
    """
    Postprocess the image to map to 0 to 1.

    Args:
        x: with shape (B, H, W), pixel values in [-1, 1]

    Returns:
        with shape (B, H, W), pixel values in [0, 1]
    """
    return (x + 1.0) / 2.0


class ForwardDiffuser:

    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule

    def diffuse(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: with shape (B, H, W)
            t: with shape (B,)

        Returns:
            Noisy version of x at time step t.
        """
        alphas_cumprod_t = self.schedule.alphas_cumprod[t].view(-1, 1, 1)
        noise = torch.randn_like(x)
        noisy_x = (
            torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise
        )
        return noisy_x, noise


class ReverseDiffuser:

    def __init__(self, model: torch.nn.Module, schedule: DiffusionSchedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def denoise(self, x: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        """
        Args:
            x: with shape (B, H, W)

        Returns:
            Denoised version of noisy_x at time step t.
        """

        t_tensor = torch.zeros(
            (x.shape[0],),
            device=x.device,
            dtype=torch.long,
        )

        for t in range(self.schedule.num_steps - 1, -1, -1):
            alpha_t = self.schedule.alphas[t]
            alpha_cum_t = self.schedule.alphas_cumprod[t]
            beta_t = self.schedule.betas[t]
            t_tensor.fill_(t)

            pred_noise = self.model(x, t_tensor)[..., 0]
            x = (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_cum_t) * pred_noise
            ) / torch.sqrt(alpha_t)
            x += torch.sqrt(beta_t) * torch.randn_like(x)
            yield x
