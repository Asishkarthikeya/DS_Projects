# projection_heads.py

import torch
import torch.nn as nn

class VisualProjectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = nn.LayerNorm(config.embed_dim)
        self.projection = nn.Linear(config.embed_dim, config.embed_dim)
        self.output_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Visual features (e.g., patch embeddings or global features)
                              Shape (B, NumPatches, input_dim) or (B, input_dim).
        Returns:
            torch.Tensor: Projected and normalized features, 
                          shape (B, NumPatches, output_dim) or (B, output_dim).
        """
        x = self.input_norm(x)  # Normalize input features
        x = self.projection(x)  # Linear projection
        x = self.output_norm(x)  # Normalize output features
        return x

class LanguageProjectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = nn.LayerNorm(config.embed_dim)
        self.projection = nn.Linear(config.embed_dim, config.embed_dim)
        self.output_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Textual features (e.g., token embeddings or global features)
                              Shape (B, SeqLen, input_dim) or (B, input_dim).
        Returns:
            torch.Tensor: Projected and normalized features, 
                          shape (B, SeqLen, output_dim) or (B, output_dim).
        """
        x = self.input_norm(x)  # Normalize input features
        x = self.projection(x)  # Linear projection
        x = self.output_norm(x)  # Normalize output features
        return x