# mlrg/src/modules/multimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusionLayer(nn.Module):
    """ Single layer of the Multi-modal Fusion network.
        Resembles a Transformer Decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.dropout = config.dropout

        # Layer 1: Self-Attention on textual priors + Add & Norm
        self.self_attn_norm = nn.LayerNorm(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        self.dropout_sa = nn.Dropout(self.dropout)

        # Layer 2: Cross-Attention (Textual priors attend to Visual features) + Add & Norm
        self.cross_attn_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        self.dropout_ca = nn.Dropout(self.dropout)

        # Layer 3: Feedforward + Add & Norm
        self.ffn_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, text_features, visual_features, text_padding_mask=None, visual_padding_mask=None):
        """
        Args:
            text_features (torch.Tensor): Input textual features (e.g., from encoded priors), shape (B, T, D).
            visual_features (torch.Tensor): Input visual features (e.g., from MLF), shape (B, P, D).
            text_padding_mask (torch.Tensor, optional): Mask for text sequence (B, T). True indicates padding.
            visual_padding_mask (torch.Tensor, optional): Mask for visual sequence (B, P). True indicates padding.

        Returns:
            torch.Tensor: Fused textual features, shape (B, T, D).
        """
        # ----- Self Attention on Text -----
        residual_sa = text_features
        text_norm = self.self_attn_norm(text_features)
        # Self attention: Q=K=V=text_norm
        sa_output, _ = self.self_attn(
            query=text_norm,
            key=text_norm,
            value=text_norm,
            key_padding_mask=text_padding_mask,
            need_weights=False
        )
        text_features = residual_sa + self.dropout_sa(sa_output) # Add & Norm implicitly done by LayerNorm + residual

        # ----- Cross Attention (Text attends to Vision) -----
        residual_ca = text_features
        text_norm = self.cross_attn_norm(text_features) # Norm before cross-attn query
        # visual_features might need norm if not already applied
        visual_norm = self.cross_attn_norm(visual_features) # Re-use norm layer? Or separate? Fig 3B doesn't show norm on visual input here. Assume input is normalized.

        # Cross attention: Q=text_norm, K=V=visual_features
        ca_output, _ = self.cross_attn(
            query=text_norm,
            key=visual_features, # Use potentially un-normed visual features as K,V if that's intended
            value=visual_features,
            key_padding_mask=visual_padding_mask,
            need_weights=False
        )
        text_features = residual_ca + self.dropout_ca(ca_output) # Add & Norm

        # ----- Feed Forward -----
        residual_ffn = text_features
        text_norm = self.ffn_norm(text_features)
        ffn_output = self.mlp(text_norm)
        fused_features = residual_ffn + ffn_output # Add & Norm

        return fused_features

class MultiModalFusionNetwork(nn.Module):
    """ Multi-modal Fusion Network (Stack of MultiModalFusionLayers)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers=config.mmf_num_layers
        self.embed_dim=config.embed_dim
        self.num_heads=config.num_heads
        self.dropout=config.dropout

        self.logger = config.get_logger(__name__)

        self.logger.info(f"Initializing Multi-modal Fusion Network with {self.num_layers} layers.")
        self.layers = nn.ModuleList([
            MultiModalFusionLayer(self.config) for _ in range(self.num_layers)
        ])

    def forward(self, text_features, visual_features, text_padding_mask=None, visual_padding_mask=None):
        """
        Processes features through the stack of MultiModalFusion layers.

        Args:
            text_features (torch.Tensor): Input textual features (B, T, D).
            visual_features (torch.Tensor): Input visual features (B, P, D).
            text_padding_mask (torch.Tensor, optional): Padding mask for text (B, T).
            visual_padding_mask (torch.Tensor, optional): Padding mask for vision (B, P).

        Returns:
            torch.Tensor: Fused textual features (B, T, D).
        """
        output = text_features
        for layer in self.layers:
            output = layer(
                text_features=output, # Pass output of previous layer as input text features
                visual_features=visual_features, # Visual features might be constant input across layers
                text_padding_mask=text_padding_mask,
                visual_padding_mask=visual_padding_mask
            )
        return output