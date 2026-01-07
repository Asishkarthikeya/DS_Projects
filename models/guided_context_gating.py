#### Guided Context Gating Attention ####
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalContextAttention(nn.Module):
    def __init__(self, filters, reduction_ratio=8, kernel = 1, transform_output_activation='linear'):
        super(GlobalContextAttention, self).__init__()
        self.channels = filters
        self.kernel = kernel

        # Context Formulation block
        self.context_conv = nn.Conv2d(self.channels, 1, kernel_size=self.kernel, stride=1, padding=0, bias=False)

        # Channel Correlation / Transform block
        self.transform_bottleneck = nn.Conv2d(self.channels, self.channels // reduction_ratio, kernel_size=self.kernel, stride=1, padding=0, bias=False)
        self.transform_activation = nn.ReLU(inplace=True)
        
        self.layer_norm = nn.LayerNorm([self.channels // reduction_ratio])
    
        self.transform_conv = nn.Conv2d(self.channels // reduction_ratio, self.channels, kernel_size=self.kernel, stride=1, padding=0, bias=False)
        self.transform_output_activation = nn.Identity(self.channels)
        
        # Module's output
        self.output = None


    def forward(self, x):
        batch_size = x.size(0)

        # Context Formulation block
        input_context_1 = x.view(x.size(0), x.size(1), -1)   # (B, C, H*W)
        
        input_context_2 = self.context_conv(x)  # (B, 1, H, W)
        input_context_2 = F.softmax(input_context_2, dim=-1)   # (B, 1, H, W)
        # Reshape to (B, H*W, 1)
        batch, _, height, width = input_context_2.size()
        input_context_2 = input_context_2.view(batch, height*width, 1)   # (B, H*W, 1)

        # Compute context block outputs
        context = torch.bmm(input_context_1, input_context_2)  # (B, C, 1)
        context = context.view(context.size(0), context.size(1), -1, 1) # (B, C, 1, 1)

        # Channel Correlation / Transform block
        transform = self.transform_bottleneck(context)
        transform = self.transform_activation(transform)  # (B, C/k, 1, 1)
        transform = transform.squeeze(-1).squeeze(-1)     # (B, C/k)
        transform = self.layer_norm(transform)
        transform = transform.view(transform.size(0), transform.size(1), 1, 1)   # (B, C/k, 1, 1)
        transform = self.transform_conv(transform)        # (B, C, 1, 1)
        transform = self.transform_output_activation(transform)

        # Apply context transform
        self.output = x + transform    # (B, C, H, W)
        return self.output

class AttentionGate(nn.Module):
    def __init__(self, filters):
        super(AttentionGate, self).__init__()
        
        self.conv_x = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, padding='same', bias=False)
        self.conv_g = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, padding='same', bias=False)
        self.psi = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=1, stride=1, padding='same', bias=False)
        self.layer_norm = nn.LayerNorm(filters)
        self.bxg = nn.Parameter(torch.zeros(filters))
        self.bpsi = nn.Parameter(torch.zeros(1))
        
        # Module's output
        self.output = None

    def forward(self, x, g):
        # Apply convolutional operations
        x_conv = self.conv_x(x)
        g_conv = self.conv_g(g)

        # Compute additive attention
        att = F.relu(x_conv + g_conv + self.bxg.view(1, -1, 1, 1))
        
        att = att.permute(0, 2, 3, 1)   # Permuting (B, H, W, C) 'cause layer_norm expects last dimension to be C
        att = self.layer_norm(att)
        att = att.permute(0, 3, 1, 2)   # Permuting back to (B, C, H, W) as PyTorch expects
        att = self.psi(att) + self.bpsi.view(1, -1, 1, 1)
        att = torch.sigmoid(att)

        self.output = att * x
        return self.output