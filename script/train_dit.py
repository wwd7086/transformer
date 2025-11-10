import os
from dataclasses import asdict
from datetime import datetime

import torch
import torch.nn.functional as F

import dataset.mnist.data as data
import model.dit as dit
from model.diffusion_schedule import get_diffusion_schedule
from model.diffusion import ForwardDiffuser, preprocess_image

# TODO:
# 1. Add class conditioning
# 1.1 Add classifer free guidance
# 2. Add consine based diffusion schedule
# 3. Add predicted variance to the model
# 6. Add moving average model weights
# 7. (Not working) use beta distribution for time embedding
# 9. Finetune the model intilization with reference implementation.

# Dataset
dataset_name = "mnist"
data_dir = os.path.join("dataset", dataset_name)

# Output
now = datetime.now()
run_name = now.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", "dit", run_name)
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
batch_size = 64
max_iters = 5000
learning_rate = 1e-4
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
weight_decay = 0.0

print_interval = 50
save_interval = 500

num_diffusion_steps = 1000
beta_start = 0.0001
beta_end = 0.02

dit_config = dit.TinyDiTConfig(
    image_dim=28,
    patch_size=4,
    emb_dim=256,
    output_dim=1,
    num_layers=6,
    num_heads=8,
    query_group_size=1,
    context_length=49,
    time_max_period=num_diffusion_steps,
    time_num_thetas=32,
    time_output_dim=256,
)

# Initialize the model.
dit_model = dit.TinyDiT(dit_config)

diffusion_schedule = get_diffusion_schedule(
    timesteps=num_diffusion_steps, beta_start=beta_start, beta_end=beta_end
)
forward_diffuser = ForwardDiffuser(diffusion_schedule)


# Intialize the optimizer.
optimizer = dit_model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
)


def train_loop():
    for iter in range(max_iters):
        # Fetch data.
        x, y = data.get_batch(
            data_dir,
            "train",
            batch_size,
            filter_label=8,
        )

        # Sample noise x.
        sampled_t = torch.randint(
            0, diffusion_schedule.num_steps, (x.shape[0],), device=x.device
        )
        # preprocess the image to map to -1 to 1.
        x = preprocess_image(x)
        noise_x, sampled_noise = forward_diffuser.diffuse(x, sampled_t)
        # Forward and get loss.
        # TODO: Incorporate time step t information into the model.
        pred_noise = dit_model(noise_x, sampled_t)
        loss = F.mse_loss(pred_noise[..., 0], sampled_noise)

        # Graident and update.
        optimizer.zero_grad()
        loss.backward()
        # Clip the graidents
        torch.nn.utils.clip_grad_norm_(dit_model.parameters(), grad_clip)
        optimizer.step()

        if iter % print_interval == 0:
            print(f"Step: {iter}, Loss: {loss.item():.4f}")

        if iter % save_interval == 0 or iter == max_iters - 1:
            # Save checkpoint
            checkpoint = {
                "step": iter,
                "model_state_dict": dit_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "config": asdict(dit_config),
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, f"ckpt_{iter}.pth"),
            )
            print(f"Step: {iter}, saved ckpt")


if __name__ == "__main__":
    train_loop()
