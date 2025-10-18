import torch.nn as nn
import torch


class TokenEmbedder(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()

        # TODO: Might need to clip the values.
        self.embedding = nn.Parameter(torch.randn((vocab_size, emb_dim)))

    def encode(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tokens: with shape [..., ]
            Note: The index value should fall within vocab_size
        Returns:
            with shape [..., emb_dim]
        """
        return self.embedding[input_tokens, :]

    def decode(self, input_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_emb: with shape [..., emb_dim]

        Returns:
            with shape [..., vocab_size]
        """

        return input_emb @ self.embedding.transpose(1, 0)


if __name__ == "__main__":
    vocab_size = 8
    emb_dim = 32

    embedder = TokenEmbedder(vocab_size, emb_dim)
    test_input = torch.arange(0, 8, dtype=torch.int64)
    test_input = test_input.reshape(2, 4)

    input_emb = embedder.encode(test_input)
    assert input_emb.shape == (2, 4, emb_dim)

    input_decode = embedder.decode(input_emb)
    assert input_decode.shape == (2, 4, vocab_size)
    # Argmax
    input_decode_max = torch.argmax(input_decode, dim=-1)
    assert torch.all(input_decode_max == test_input)
    print(input_decode_max)
