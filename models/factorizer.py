import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TemoporalFactorizer(nn.Module):
    def __init__(self, config):
        super(TemoporalFactorizer, self).__init__()
        self.config = config
        self.hidden_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_layers = config.fact_num_layers
        
        # Shared Encoder: Transformer for parallelism
        encoder_layer = TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=self.hidden_dim*4)
        self.shared_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Specific Encoders: Separate Transformers for current/prior
        self.current_specific_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.prior_specific_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def forward(self, current_embed, prior_embed):
        """
        current_embed: [B, L, D]
        prior_embed:   [B, L, D]
        """
        # Parallel Transformer processing for shared features
        shared_current = self.shared_encoder(current_embed)
        shared_prior = self.shared_encoder(prior_embed)
        
        # Specific features with parallelism
        specific_current = self.current_specific_encoder(current_embed)
        specific_prior = self.prior_specific_encoder(prior_embed)
        
        return shared_current, shared_prior, specific_current, specific_prior