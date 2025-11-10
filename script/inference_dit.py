import os

import torch
import matplotlib.pyplot as plt

import model.dit as dit
from model.diffusion import ReverseDiffuser, postprocess_image
from model.diffusion_schedule import get_diffusion_schedule

dataset_name = "mnist"
data_dir = os.path.join("dataset", dataset_name)

run_name = "20251109_144345"
ckpt_name = "ckpt_4999.pth"
output_dir = os.path.join("output", "dit", run_name, ckpt_name)

# Initialize the model.
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

dit_model = dit.TinyDiT(dit_config)
dit_model.eval()

# Load the weights.
model_states = torch.load(output_dir)
dit_model.load_state_dict(model_states["model_state_dict"])


diffusion_schedule = get_diffusion_schedule(
    timesteps=num_diffusion_steps, beta_start=beta_start, beta_end=beta_end
)
reverse_diffuser = ReverseDiffuser(dit_model, diffusion_schedule)

# Generate samples.
num_samples = 16
sampled_noise = torch.randn(
    num_samples, dit_config.image_dim, dit_config.image_dim
)  # (B, H, W)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for step, generated_image in enumerate(reverse_diffuser.denoise(sampled_noise)):
    if step % 10 == 0 or step == diffusion_schedule.num_steps - 1:
        postprocessed_imgs = postprocess_image(generated_image)
        for i in range(num_samples):
            ax = axes[i // 4, i % 4]
            ax.clear()
            ax.imshow(postprocessed_imgs[i], cmap="gray")
            ax.axis("off")
        plt.suptitle(f"Denoising Step {step}")
        plt.pause(0.01)
plt.show()
