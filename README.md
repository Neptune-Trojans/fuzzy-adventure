# Self-Attention

## Usage

```python
import torch
from attention.self_attention import SelfAttention

# Create model
model = SelfAttention(input_dim=64)

# Input shape: (batch_size, seq_length, input_dim)
x = torch.randn(4, 10, 64)

# Forward pass
output = model(x)  # Output shape: (4, 10, 64)
```

## Run Test

```bash
python attention/self_attention.py
```

## Architecture

![Self-Attention Mechanism](images/self_attention.png)

## Reference

Vaswani et al. (2017) - "Attention Is All You Need"
https://arxiv.org/abs/1706.03762