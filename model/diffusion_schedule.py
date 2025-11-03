# Generate the diffusion schedule.
# alpha, beta and accumulated alpha.

from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    num_steps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor


def linear_beta_schedule(
    timesteps: int, beta_start: float, beta_end: float
) -> torch.Tensor:
    """
    Generate a linear beta schedule.

    Args:
        timesteps: number of diffusion steps
        beta_start: starting value of beta
        beta_end: ending value of beta

    Returns:
        A tensor of shape (timesteps,) containing the beta schedule.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_schedule(
    timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """
    Create the diffusion schedule.

    Args:
        timesteps: number of diffusion steps
        beta_start: starting value of beta
        beta_end: ending value of beta

    Returns:
        An instance of DiffusionSchedule containing betas, alphas, and alphas_cumprod.
    """
    betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return DiffusionSchedule(
        num_steps=timesteps,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
    )


if __name__ == "__main__":
    schedule = get_diffusion_schedule(timesteps=1000)

    # plot the schedules
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(schedule.betas.numpy())
    plt.title("Beta Schedule")
    plt.subplot(1, 3, 2)
    plt.plot(schedule.alphas.numpy())
    plt.title("Alpha Schedule")
    plt.subplot(1, 3, 3)
    plt.plot(schedule.alphas_cumprod.numpy())
    plt.title("Alpha Cumprod Schedule")
    plt.show()
