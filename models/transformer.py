"""
This module contains a minimal Transformer class for sequence tasks and a demo function.
"""

import torch
from torch import nn

from models.base import BaseModel


class Transformer(BaseModel):
    """
    Minimal decoder-only Transformer model for sequence tasks.

    NOTE: the "decoder" here refers to the GPT-style architecture where
    causal masking is applied in the self-attention layers. The implementation
    uses `nn.TransformerEncoder` with a causal mask to achieve this.
    """

    def __init__(
        self,
        num_digits: int = 3,
        embed_dim: int = 128,
        max_seq_len: int = 5000,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        transformer_num_layers: int = 6,
        **_kwargs,
    ):
        """
        Initializes the Transformer model.

        Args:
            num_digits: number of unique tokens (vocabulary size)
            embed_dim: embedding dimension
            max_seq_len: maximum sequence length
            num_heads: number of attention heads
            dropout_rate: dropout rate (float in range [0, 1])
            transformer_num_layers: number of transformer layers
        """
        super().__init__()

        self.num_digits = num_digits

        self.token_embedding = nn.Embedding(num_digits, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_num_layers, enable_nested_tensor=False
        )

        self.fc = nn.Linear(embed_dim, num_digits)

        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes model parameters.

        - Linear layers: weights ~ N(0, 0.02), biases = 0
        - Embeddings: weights ~ N(0, 0.02)
        - LayerNorm: bias = 0, weight = 1
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        Args:
            x: input batch of shape (batch_size, seq_len)
                - int values in range [0, num_digits-1]

        Returns:
             predicted logits of shape (batch_size, seq_len, num_digits)
        """
        _, seq_len = x.size()
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )

        positions = torch.arange(
            0, seq_len, device=x.device, dtype=torch.long
        ).unsqueeze(0)

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )

        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.layernorm(x)

        logits = self.fc(x)

        return logits

    def generate(
        self, x: torch.Tensor, max_length: int = 20, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate sequences using the transformer model.

        Args:
            x: input batch of shape (batch_size, seq_len)
            max_length: number of tokens to generate
            temperature: sampling temperature

        Returns:
            generated sequences of shape (batch_size, seq_len + max_length)
        """
        for _ in range(max_length):
            logits = self(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            _, next_token = torch.topk(probs, k=1, dim=-1)
            x = torch.cat([x, next_token], dim=1)
        return x


def demo():
    """
    Demonstrates the forward pass of the model with a random input tensor.

    This function:
    - Initializes the Transformer model
    - Runs a forward pass on a random input tensor
    - Prints model parameter count and input/output shapes
    """

    batch_size = 4
    seq_len = 10
    num_digits = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(num_digits=num_digits).to(device)

    x = torch.randint(0, num_digits, (batch_size, seq_len), device=device)
    y = model(x)

    print(f"Model: {model.__class__.__name__}")
    print(f"├── # params.......: {model.n_param / 1e6:.2f}M")
    print(f"├── batch size.....: {batch_size}")
    print(f"├── input shape....: {tuple(x.shape)}")
    print(f"└── output shape...: {tuple(y.shape)}")


if __name__ == "__main__":
    demo()
