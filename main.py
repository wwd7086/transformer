from transformer import Transformer
import torch


def main():
    print("Hello from transformer!")

    B, T, C = 2, 8, 16

    transformer = Transformer(feature_dim=C, num_heads=2, context_length=T)

    test_input = torch.zeros(B, T, C)

    test_output = transformer(test_input)

    print(test_output.shape)


if __name__ == "__main__":
    main()
