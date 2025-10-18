import torch
import torch.nn.functional as F


def token_classification_loss(
    pred_token_logits: torch.Tensor,
    gt_token_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        pred_token_logits: with shape [...., vocab_size]
        gt_token_idx: with shape [...], type int64
    """
    loss = F.cross_entropy(
        pred_token_logits.flatten(start_dim=0, end_dim=-2),
        gt_token_idx.flatten(),
        ignore_index=-1,
        reduction="mean",
    )
    return loss


if __name__ == "__main__":
    B, T, C = 4, 8, 16
    test_pred = torch.randn((B, T, C))
    test_gt = torch.randint(C, (B, T))

    loss = token_classification_loss(test_pred, test_gt)
    assert loss.shape == ()
    print(loss)
