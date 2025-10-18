import os
import math
from dataclasses import asdict
from datetime import datetime

import torch

import data
import gpt
import loss

# Dataset
dataset_name = "shakespeare_char"
data_dir = os.path.join("dataset", dataset_name)
meta_path = os.path.join(data_dir, "meta.pkl")
vocab_size = data.get_vocab_size(meta_path)
assert vocab_size is not None

# Output
now = datetime.now()
run_name = now.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", run_name)
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
batch_size = 16
max_iters = 5000
warmup_iters = 20  # not super necessary potentially
lr_decay_iters = 5000  # make equal to max_iters usually
learning_rate = 1e-3  # with baby networks can afford to go a bit higher
min_lr = 1e-4  # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
weight_decay = 3e-2

print_interval = 50
save_interval = 500
eval_interval = 500
eval_iters = 200

gpt_config = gpt.TinyGPTConfig(
    emb_dim=256,
    num_layers=6,
    num_heads=8,
    query_group_size=2,
    vocab_size=vocab_size,
    context_length=256,
)

# Initialize the model.
gpt_model = gpt.TinyGPT(gpt_config)


# Initialize the LR schedule.
def get_lr(it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss() -> dict:
    out = {}
    gpt_model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(
                data_dir,
                split,
                gpt_config.context_length,
                batch_size,
            )
            y_pred = gpt_model(x)
            cls_loss = loss.token_classification_loss(y_pred, y)
            losses[k] = cls_loss.item()
        out[split] = losses.mean()
    gpt_model.train()
    return out


# Intialize the optimizer.
optimizer = gpt_model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
)

# Figure out network intialization
# Model summary logging


def train_loop():
    for iter in range(max_iters):
        # Set learning rate.
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Fetch data.
        x, y = data.get_batch(
            data_dir,
            "train",
            gpt_config.context_length,
            batch_size,
        )

        # Forward and get loss.
        y_pred = gpt_model(x)
        cls_loss = loss.token_classification_loss(y_pred, y)

        # Graident and update.
        optimizer.zero_grad()
        cls_loss.backward()
        # Clip the graidents
        torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), grad_clip)
        optimizer.step()

        if iter % print_interval == 0:
            print(f"Step: {iter}, Loss: {cls_loss.item():.4f}")

        if iter % save_interval == 0 or iter == max_iters - 1:
            # Save checkpoint
            checkpoint = {
                "step": iter,
                "model_state_dict": gpt_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": cls_loss.item(),
                "config": asdict(gpt_config),
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, f"ckpt_{iter}.pth"),
            )
            print(f"Step: {iter}, saved ckpt")

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"Step: {iter}, train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}"
            )


if __name__ == "__main__":
    train_loop()
