#!/usr/bin/env python3
"""UNet3D Backbone Implementation.

Implements 3D UNet architecture for diffusion model backbone including
encoder, decoder, attention mechanisms, and skip connections for
morphogen concentration prediction.

Integration: Neural network backbone for ML diffusion system
Rationale: Focused UNet3D architecture separated from training logic
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .ml_diffusion_types import UNetArchitecture

logger = logging.getLogger(__name__)

class Conv3DBlock(nn.Module):
    """3D convolution block with normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1):
        """Initialize 3D convolution block."""
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return x

class AttentionBlock3D(nn.Module):
    """3D attention block for UNet."""
    
    def __init__(self, channels: int):
        """Initialize 3D attention block."""
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention block."""
        B, C, D, H, W = x.shape
        
        # Normalize input
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention computation
        q = q.view(B, C, -1).transpose(1, 2)  # B, DHW, C
        k = k.view(B, C, -1)                  # B, C, DHW
        v = v.view(B, C, -1).transpose(1, 2)  # B, DHW, C
        
        # Attention weights
        scale = (C ** -0.5)
        attn = torch.softmax(q @ k * scale, dim=-1)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).view(B, C, D, H, W)
        
        # Output projection
        out = self.out_proj(out)
        
        return x + out

class UNet3DBackbone(nn.Module):
    """3D UNet backbone for diffusion model.
    
    Implements 3D UNet architecture with encoder-decoder structure,
    skip connections, and optional attention mechanisms for morphogen
    concentration prediction and enhancement.
    """
    
    def __init__(self, input_channels: int = 4, output_channels: int = 4,
                 base_channels: int = 64, architecture: UNetArchitecture = UNetArchitecture.STANDARD):
        """Initialize UNet3D backbone.
        
        Args:
            input_channels: Number of input channels (morphogens)
            output_channels: Number of output channels
            base_channels: Base number of channels
            architecture: UNet architecture variant
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.architecture = architecture
        
        # Channel progression
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            Conv3DBlock(input_channels, channels[0]),
            Conv3DBlock(channels[0], channels[1]),
            Conv3DBlock(channels[1], channels[2]),
            Conv3DBlock(channels[2], channels[3])
        ])
        
        # Attention blocks (if attention architecture)
        if architecture == UNetArchitecture.ATTENTION:
            self.attention_blocks = nn.ModuleList([
                AttentionBlock3D(channels[i]) for i in range(len(channels))
            ])
        else:
            self.attention_blocks = None
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Conv3DBlock(channels[3] + channels[2], channels[2]),
            Conv3DBlock(channels[2] + channels[1], channels[1]),
            Conv3DBlock(channels[1] + channels[0], channels[0]),
            Conv3DBlock(channels[0], channels[0])
        ])
        
        # Output projection
        self.output_conv = nn.Conv3d(channels[0], output_channels, 1)
        
        # Downsampling and upsampling
        self.downsample = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        logger.info(f"Initialized UNet3D with {architecture.value} architecture")
        logger.info(f"Channels: {input_channels} → {output_channels}, base: {base_channels}")
    
    def forward(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through UNet3D.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            timestep: Diffusion timestep (optional)
            
        Returns:
            Output tensor (B, C, D, H, W)
        """
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            
            # Apply attention if available
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
            
            skip_connections.append(x)
            
            # Downsample (except for last block)
            if i < len(self.encoder_blocks) - 1:
                x = self.downsample(x)
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample (except for first block)
            if i > 0:
                x = self.upsample(x)
                
                # Ensure spatial dimensions match for skip connection
                skip = skip_connections[-(i+1)]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
                
                # Concatenate skip connection
                x = torch.cat([x, skip], dim=1)
            
            x = decoder_block(x)
        
        # Output projection
        output = self.output_conv(x)
        
        return output
    
    def get_model_summary(self) -> Dict[str, int]:
        """Get model architecture summary.
        
        Returns:
            Dictionary with model statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_blocks": len(self.encoder_blocks),
            "decoder_blocks": len(self.decoder_blocks),
            "has_attention": self.attention_blocks is not None,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels
        }
