import timm
import torch.nn as nn
from models.guided_context_gating import GlobalContextAttention, AttentionGate

class CNNEncoder(nn.Module):
    def __init__(self, config):
        super(CNNEncoder, self).__init__()

        self.config = config
        self.logger = config.get_logger(__name__)
        self.logger.info(f"Initializing CNN Encoder: {config.cnn_model_name}")

        # Load the pre-trained EfficientNetB0 model
        self.model = timm.create_model(config.cnn_model_name, pretrained=True, num_classes=0)
        
        # Remove the fully connected layer (last layer)
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        # Dynamically find the last convolutional layer
        last_conv_layer = self._get_last_conv_layer(self.model)
        
        # Get the output channels of the last convolutional layer
        conv_output_channels = last_conv_layer.out_channels

        # Global Context Attention
        self.gc_attention = GlobalContextAttention(filters=conv_output_channels)
        
        # Attention Gate Block
        self.attention_gate = AttentionGate(filters=conv_output_channels)
        
        # Define the final fully connected layer for embedding
        self.fc = nn.Linear(conv_output_channels, config.embed_dim)

        # Optionally freeze the encoder layers
        if config.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Pass through the convolutional part of the model to get features
        features = self.model(x)  # (Batch, C, H, W)

        # Apply Global Context Attention
        gc_features = self.gc_attention(features)
        att_features = self.attention_gate(features, gc_features)
        
        # Reshape to (batch_size, H * W, channels)
        batch_size, channels, height, width = att_features.size()   
        att_features = att_features.view(batch_size, -1, channels,)  # (Batch, H*W, C)
        
        # Optionally pass through the final embedding layer to reduce dimensionality
        image_features = self.fc(att_features)  # (Batch, H*W, embed_dim)
        
        return image_features

    def _get_last_conv_layer(self, model):
        """
        Helper method to get the last convolutional layer in the ResNet model.
        It iterates through the blocks and returns the final convolutional layer.
        """
        # ResNet is composed of several blocks of layers
        for layer in model:
            if isinstance(layer, nn.Conv2d):  # Check if the layer is a Conv2d layer
                last_conv_layer = layer
        return last_conv_layer