# longitudinal_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LongitudinalFusionLayer(nn.Module):
    """ Single layer of the Multi-view Longitudinal Fusion network (Fig 3A)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim=config.embed_dim
        self.num_heads=config.num_heads
        self.dropout=config.dropout

        # Layer 1: Cross-Attention related Norm + Attention + Residual
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout)

        # Layer 2: Feedforward related Norm + MLP + Residual
        self.norm2 = nn.LayerNorm(self.embed_dim)
        # Standard Transformer MLP: Linear -> ReLU/GELU -> Dropout -> Linear -> Dropout
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(), # Or ReLU
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, query, key_value, key_padding_mask=None):
        """
        Args:
            query (torch.Tensor): Anchor scan features (B, P, D) where P is num_patches.
                                  In the paper's Eq 5, it's v_ia^cur (P, D) for one sample.
                                  When batched, needs careful handling.
            key_value (torch.Tensor): Concatenated auxiliary refs + prev image features (B, N_kv, P, D).
                                      This also needs careful reshaping/masking per sample.
                                      Or (B, N_kv*P, D) if flattened.
            key_padding_mask (torch.Tensor, optional): Mask for key_value sequence (B, N_kv*P).

        Returns:
            torch.Tensor: Fused features (B, P, D).
        """
        # ----- Cross Attention Block -----
        # Eq 5 processes one sample. For batching:
        # query: (B, P, D) - features of anchor scan for each sample
        # key_value: (B, N_kv*P, D) - flattened features of aux+prev for each sample
        # We assume input features are already projected (D = cfg.PROJECTION_DIM)

        # Residual connection 1 start
        residual1 = query

        # Pre-Normalization (common practice)
        query_norm = self.norm1(query)
        # key_value might need its own norm if not already normalized
        kv_norm = self.norm1(key_value) # Applying same norm, maybe should be separate? Fig 3A implies norm before CrossAttn.

        # Cross Attention (Query attends to Key/Value)
        # query=Q, key=K, value=V
        attn_output, _ = self.cross_attn(
            query=query_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask, # Mask out padding tokens in the K/V sequence
            need_weights=False # Don't need attention weights unless for analysis
        )
        # Dropout and Residual connection 1 end
        x = residual1 + self.dropout1(attn_output)

        # ----- Feed Forward Block -----
        # Residual connection 2 start
        residual2 = x

        # Pre-Normalization
        x_norm = self.norm2(x)

        # MLP
        mlp_output = self.mlp(x_norm)

        # Residual connection 2 end
        fused_features = residual2 + mlp_output

        return fused_features

class LongitudinalFusionNetwork(nn.Module):
    """ Longitudinal Fusion Network (Stack of LFLayers)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers=config.lf_num_layers
        self.embed_dim=config.embed_dim
        self.num_heads=config.num_heads
        self.dropout=config.dropout
        self.logger = config.get_logger(__name__)
        self.logger.info(f"Initializing LF Network with {self.num_layers} layers.")
        self.layers = nn.ModuleList([
            LongitudinalFusionLayer(self.config) for _ in range(self.num_layers)
        ])

    def forward(self, anchor_features, context_features, context_mask=None):
        """
        Processes features through the stack of MLF layers.

        Args:
            anchor_features (torch.Tensor): Query features (B, P_anchor, D).
            context_features (torch.Tensor): Key/Value context features (B, P_context, D).
            context_mask (torch.Tensor, optional): Padding mask for context (B, P_context). True indicates padding.

        Returns:
            torch.Tensor: Fused features (B, P_anchor, D).
        """
        output = anchor_features
        for layer in self.layers:
            output = layer(output, context_features, key_padding_mask=context_mask)
        return output