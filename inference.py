import os

import torch

import gpt
import data

dataset_name = "shakespeare_char"
data_dir = os.path.join("dataset", dataset_name)
meta_path = os.path.join(data_dir, "meta.pkl")
vocab_size = data.get_vocab_size(meta_path)
encode, decode = data.get_token_enc_dec(meta_path)
assert vocab_size is not None

run_name = "20250921_141714"
ckpt_name = "ckpt_4999.pth"
output_dir = os.path.join("output", run_name, ckpt_name)

# Initialize the model.
gpt_config = gpt.TinyGPTConfig(
    emb_dim=192,
    num_layers=6,
    num_heads=6,
    vocab_size=vocab_size,
    context_length=128,
)
gpt_model = gpt.TinyGPT(gpt_config)
gpt_model.eval()

# Load the weights.
model_states = torch.load(output_dir)
gpt_model.load_state_dict(model_states["model_state_dict"])

# Test autoregressive decoding.
print("\n Autoregressive Inference:")
print("\n---------")

prompt = "Fear yo"
print(f"\033[31m{prompt}\033[0m", end="")
prompt_idx = torch.tensor(encode(prompt), dtype=torch.int64)[None, ...]
for out_idx in gpt_model.sample(prompt_idx, max_step=5000, temperature=1.0):
    print(decode(out_idx.tolist()[0]), end="")
