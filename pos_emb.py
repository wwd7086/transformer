import torch


def gen_thetas(num_thetas: int, max_period: float = 10000) -> torch.Tensor:
    """
    Return:
        with shape (N,)
    """
    theta_ids = torch.arange(0, num_thetas) / float(num_thetas)
    return torch.pow(max_period, -theta_ids)


def gen_sin_emb(thetas: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Return:
        with shape (T, 2N, ),
        whre: T - number steps
        eg: [[sin(0*t1), sin(0*t1), sin(0*t2), sin(0*t2), ....],
             [sin(1*t1), sin(1*t1), sin(1*t2), sin(1*t2), ....],
             ...
            ]
    """
    step_ids = torch.arange(0, num_steps, dtype=torch.float)
    sin_emb = torch.sin(step_ids[:, None] @ thetas[None, :])
    return torch.repeat_interleave(sin_emb, 2, dim=1)


def gen_cos_emb(thetas: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Return:
        with shape (T, 2N, ),
        whre: T - number steps
        eg: [[cos(0*t1), cos(0*t1), cos(0*t2), cos(0*t2), ....],
             [cos(1*t1), cos(1*t1), cos(1*t2), cos(1*t2), ....],
             ...
            ]
    """
    step_ids = torch.arange(0, num_steps, dtype=torch.float)
    cos_emb = torch.cos(step_ids[:, None] @ thetas[None, :])
    return torch.repeat_interleave(cos_emb, 2, dim=1)


def apply_rot_emb(
    input_emb: torch.Tensor,
    sin_emb: torch.Tensor,
    cos_emb: torch.Tensor,
) -> torch.Tensor:
    """
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

    sin_emb = gen_sin_emb(thetas, num_steps)
    assert sin_emb.shape == (num_steps, 2 * num_thetas)
    print(sin_emb)

    cos_emb = gen_cos_emb(thetas, num_steps)
    assert cos_emb.shape == (num_steps, 2 * num_thetas)
    print(cos_emb)

    input_emb = torch.ones((2, 2, num_steps, num_thetas * 2), dtype=torch.float)
    input_emb[..., ::2] *= -2
    print(input_emb)
    rot_emb = apply_rot_emb(input_emb, sin_emb, cos_emb)
    assert rot_emb.shape == input_emb.shape
    print(rot_emb)
