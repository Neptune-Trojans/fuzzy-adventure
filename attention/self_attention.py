import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = F.softmax(scores, dim=2)
        weighted = torch.bmm(attention, values)
        return weighted


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters for test data
    batch_size = 4
    seq_length = 10
    input_dim = 64
    test_data = torch.randn(batch_size, seq_length, input_dim)

    print(f"Input shape: {test_data.shape}")
    print(f"Input data sample (first 3 values): {test_data[0, 0, :3]}")

    # Initialize the self-attention model
    model = SelfAttention(input_dim)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(test_data)

    print(f"\nOutput shape: {output.shape}")

    # Verify output shape matches input shape
    assert output.shape == test_data.shape, "Output shape should match input shape"
    print("\nTest passed! Output shape matches input shape.")



if __name__ == "__main__":
    main()