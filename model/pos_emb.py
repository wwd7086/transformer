import torch


def gen_thetas(num_thetas: int, max_period: float = 10000) -> torch.Tensor:
    """Generate theta samples for positional encoding.

    Return:
        with shape (N,)
    """
    theta_ids = torch.arange(0, num_thetas) / float(num_thetas)
    return torch.pow(max_period, -theta_ids)


# --- Absolute Position Embedding --- #


def gen_sin_cos_emb(thetas: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """Generate sine and cosine embeddings.

    Args:
        thetas: with shape (N,)
        steps: with shape (B,)

    Return:
        with shape (B, 2N)
    """

    sin_emb = torch.sin(steps[:, None] * thetas[None, :])
    cos_emb = torch.cos(steps[:, None] * thetas[None, :])
    time_emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B, 2N)
    return time_emb


def gen_2d_sin_cos_emb(
    num_theta: int,
    num_height_steps: int,
    num_width_steps: int,
    max_period: float = 10000,
) -> torch.Tensor:
    """Generate 2D sine and cosine embeddings.

    Args:
        num_theta: number of theta samples
        num_height_steps: number of height steps
        num_width_steps: number of width steps
        max_period: maximum period for theta generation

    Return:
        with shape (H*W, 4N)
    """
    thetas = gen_thetas(num_theta, max_period)

    height_ids = torch.arange(0, num_height_steps, dtype=torch.float32)
    width_ids = torch.arange(0, num_width_steps, dtype=torch.float32)
    grid_row_ids, grid_col_ids = torch.meshgrid(
        width_ids,
        height_ids,
        indexing="xy",
    )
    row_emb = gen_sin_cos_emb(thetas, grid_row_ids.flatten())  # (H*W, 2N)
    col_emb = gen_sin_cos_emb(thetas, grid_col_ids.flatten())  # (H*W, 2N)
    full_emb = torch.cat([row_emb, col_emb], dim=-1)  # (H*W, 4N)
    return full_emb


# --- Rotary Position Embedding --- #


def gen_inter_sin_emb(thetas: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Generate interleaved sine embeddings.

    Return:
        with shape (T, 2N, ),
        whre: T - number steps
        eg: [[sin(0*t1), sin(0*t1), sin(0*t2), sin(0*t2), ....],
             [sin(1*t1), sin(1*t1), sin(1*t2), sin(1*t2), ....],
             ...
            ]
    """
    step_ids = torch.arange(0, num_steps, dtype=torch.float32)
    sin_emb = torch.sin(step_ids[:, None] @ thetas[None, :])
    return torch.repeat_interleave(sin_emb, 2, dim=1)


def gen_inter_cos_emb(thetas: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Generate interleaved cosine embeddings.

    Return:
        with shape (T, 2N, ),
        whre: T - number steps
        eg: [[cos(0*t1), cos(0*t1), cos(0*t2), cos(0*t2), ....],
             [cos(1*t1), cos(1*t1), cos(1*t2), cos(1*t2), ....],
             ...
            ]
    """
    step_ids = torch.arange(0, num_steps, dtype=torch.float32)
    cos_emb = torch.cos(step_ids[:, None] @ thetas[None, :])
    return torch.repeat_interleave(cos_emb, 2, dim=1)


def gen_inter_sin_cos_emb(
    num_thetas: int, num_steps: int, max_period: float = 10000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate interleaved sine and cosine embeddings.

    Return:
        sin_emb: with shape (T, 2N, ),
        cos_emb: with shape (T, 2N, ),
        whre: T - number steps
    """
    thetas = gen_thetas(num_thetas, max_period)
    sin_emb = gen_inter_sin_emb(thetas, num_steps)
    cos_emb = gen_inter_cos_emb(thetas, num_steps)
    return sin_emb, cos_emb


def apply_rot_emb(
    input_emb: torch.Tensor,
    sin_emb: torch.Tensor,
    cos_emb: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding.

    Args:
        input_emb: shape (..., T, D)
        sin_emb: shape (T, D)
        cos_emb: shape (T, D)

    Returns:
        Rotated emb with shape (...., T, D)
    """

    # original emb [x1, x2, x3, x4, ....]
    # emb shuffled [-x2, x1, -x3, x3, ....]

    odd_emb = input_emb[..., 0::2]
    even_emb = input_emb[..., 1::2]
    input_emb_s = torch.stack((-even_emb, odd_emb), dim=-1)  # [..., T, D/2, 2]
    input_emb_s = input_emb_s.flatten(start_dim=-2, end_dim=-1)  # [...., T, D]
    return input_emb * cos_emb + input_emb_s * sin_emb


if __name__ == "__main__":
    num_thetas = 3
    num_steps = 3

    thetas = gen_thetas(num_thetas)
    assert thetas.shape == (num_thetas,)
    print(thetas)

    sin_emb = gen_inter_sin_emb(thetas, num_steps)
    assert sin_emb.shape == (num_steps, 2 * num_thetas)
    print(sin_emb)

    cos_emb = gen_inter_cos_emb(thetas, num_steps)
    assert cos_emb.shape == (num_steps, 2 * num_thetas)
    print(cos_emb)

    input_emb = torch.ones((2, 2, num_steps, num_thetas * 2), dtype=torch.float32)
    input_emb[..., ::2] *= -2
    print(input_emb)
    rot_emb = apply_rot_emb(input_emb, sin_emb, cos_emb)
    assert rot_emb.shape == input_emb.shape
    print(rot_emb)
