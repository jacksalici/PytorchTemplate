import torch
from torch import nn, optim
from .base import BaseModel

import torch.nn.functional as F

class Transformer(BaseModel):
    """
    Decoder-only Transformer model for sequence tasks.
    """
    
    def __init__(self, input_dim=1, num_digits = 3, embed_dim=128, max_seq_len=5000,
                 num_heads=8, ff_dim=2048, dropout_rate=0.1,
                 attention_dropout=0.1, transformer_num_layers=6, **kwargs):
        super().__init__(**kwargs)
        
        # Use embed_dim as the model dimension
        self.num_digits = num_digits
        dropout = dropout_rate
        
        self.token_embedding = nn.Embedding(num_digits, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        
        self.fc = nn.Linear(embed_dim, num_digits)  # Output should match num_digits, not input_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(embed_dim)
        
        self._init_weights()

    def _init_weights(self):
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

    def forward(self, x):
        batch_size, seq_len = x.size()
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        
        positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
        
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.layernorm(x)

        logits = self.fc(x)
        
        return logits
    
    def generate(self, x, max_length=20, temperature=1.0):
        """
        Generate sequences using the transformer model.
        Args:
            x: input tensor
            max_length: number of tokens to generate
            temperature: sampling temperature 
        """
        for _ in range(max_length):
            logits = self(x)
            logits = logits[:, -1, :] / temperature  
            probs = torch.softmax(logits, dim=-1)
            _, next_token = torch.topk(probs, k=1, dim=-1)
            x = torch.cat([x, next_token], dim=1)  
        return x
            
            