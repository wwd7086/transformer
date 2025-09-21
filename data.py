from typing import Optional
import os

import numpy as np
import torch
import pickle


def get_batch(
    data_dir: str, split: str, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # We recreate np.memmap every batch to avoid a memory leak, as per
    if split == "train":
        data = np.memmap(
            os.path.join(data_dir, "train.bin"),
            dtype=np.uint16,
            mode="r",
        )
    else:
        data = np.memmap(
            os.path.join(data_dir, "val.bin"),
            dtype=np.uint16,
            mode="r",
        )

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    return x, y


def get_vocab_size(meta_path: str) -> Optional[int]:
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    return meta_vocab_size
