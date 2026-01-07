# adaptor.py

import torch.nn as nn

class Adaptor(nn.Module):
    """
    Adaptor module to fine-tune and enhance pre-trained encoder features for downstream tasks.
    """
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.embed_dim
        self.dropout = config.dropout

        # Bottleneck projection: input -> hidden -> output
        self.proj1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.proj2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout2 = nn.Dropout(self.dropout)
        self.ln = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        """
        x: (B, SeqLen, FeatureDim) for both vision (patch tokens) or language
        """
        x_proj = self.proj1(x)
        x_proj = self.activation(x_proj)
        x_proj = self.dropout1(x_proj)
        x_proj = self.proj2(x_proj)
        x_proj = self.dropout2(x_proj)
        x_proj = self.ln(x_proj)
        return x_proj
