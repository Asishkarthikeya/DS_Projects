# vision_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel

class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = config.get_logger(__name__)
        self.model_name = config.v_enc
        self.freeze_encoder = config.freeze_encoder
        self.logger.info(f"Initializing Vision Encoder from: {self.model_name}")
        try:
            # Load the pre-trained Vision Transformer model
            # Use AutoModel to get the base model without a specific head
            self.encoder = AutoModel.from_pretrained(self.model_name)
            # Example: Determine feature dimension if not explicitly known
            # This depends on the specific model architecture. For ViT, it's often in config.hidden_size
            # For DINOv2 base, hidden size is 768. Verify this matches cfg.VISUAL_FEATURE_DIM
            if hasattr(self.encoder.config, 'hidden_size'):
                 model_dim = self.encoder.config.hidden_size
                 if model_dim != config.embed_dim:
                    self.logger.warning(f"RAD-DINO model hidden size ({model_dim}) != config embed_dim ({config.embed_dim}). Check config.")
            else:
                self.logger.warning("Could not automatically determine RAD-DINO output dimension. Assuming config.embed_dim is correct.")

        except Exception as e:
            self.logger.error(f"Failed to load Vision Encoder model '{self.model_name}': {e}")
            raise

        if self.freeze_encoder:
            self.logger.info("Freezing Vision Encoder parameters.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
             self.logger.info("Vision Encoder parameters are trainable.")

    def forward(self, images):
        """
        Forward pass through the vision encoder.

        Args:
            images (torch.Tensor): Batch of images, shape (B, C, H, W) or (B*NumViews, C, H, W).

        Returns:
            torch.Tensor: Visual features (patch embeddings), shape (B, NumPatches, FeatureDim)
                          or (B*NumViews, NumPatches, FeatureDim).
        """

        outputs = self.encoder(pixel_values=images)
        # Return the last hidden state which contains patch embeddings + class token embedding
        # Typically shape: (BatchSize, SequenceLength=NumPatches+1, HiddenSize)
        last_hidden_state = outputs.last_hidden_state

        # We usually want patch features, potentially excluding the [CLS] token
        # Check model config or typical usage (DINOv2 often uses CLS token)
        # Let's assume we use all tokens for now (patch + CLS). Shape (B, p, d1)
        # p = num_patches + 1 (if CLS token exists and is included)
        visual_features = last_hidden_state

        # Example: If RAD-DINO uses patch features directly, maybe exclude CLS:
        # visual_features = last_hidden_state[:, 1:, :] # Exclude CLS token

        return visual_features

    def train(self, mode=True):
        super().train(mode)
        # Keep the underlying transformer in eval mode if it's frozen
        # This disables dropout, etc., which is usually desired for frozen feature extractors.
        if hasattr(self, 'encoder') and self.encoder.parameters().__next__().requires_grad == False:
             self.encoder.eval()