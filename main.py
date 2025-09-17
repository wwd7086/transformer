from transformer import Transformer
import torch


def main():
    print("Hello from transformer!")

    transformer = Transformer(feature_dim=8, num_heads=2)

    B, T, C = 2, 8, 8
    test_input = torch.zeros(B, T, C)

    test_output = transformer(test_input)

    print(test_output.shape)


if __name__ == "__main__":
    main()
