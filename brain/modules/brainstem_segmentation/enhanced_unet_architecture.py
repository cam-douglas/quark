#!/usr/bin/env python3
"""
Enhanced U-Net Architecture with Attention Gates and Deep Supervision
For improved brainstem segmentation performance targeting Dice ≥0.87
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class AttentionGate3D(nn.Module):
    """3D Attention Gate for feature refinement in skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature channels in gating signal (from decoder)
            F_l: Number of feature channels in feature map (from encoder)
            F_int: Number of intermediate channels
        """
        super(AttentionGate3D, self).__init__()
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        # Transform feature map
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        # Combine and output attention coefficients
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder path
            x: Feature map from encoder path (skip connection)
        Returns:
            Refined feature map with attention
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Element-wise sum and ReLU
        psi = self.relu(g1 + x1)
        
        # Apply attention coefficients
        psi = self.psi(psi)
        
        # Multiply attention map with input features
        return x * psi


class DeepSupervisionBlock(nn.Module):
    """Deep supervision output block for intermediate layers."""
    
    def __init__(self, in_channels: int, num_classes: int, scale_factor: int):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            scale_factor: Upsampling factor to match final output size
        """
        super(DeepSupervisionBlock, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate supervision output at current scale."""
        x = self.conv(x)
        
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, 
                            mode='trilinear', align_corners=False)
        
        return x


class EnhancedUNet3D(nn.Module):
    """Enhanced 3D U-Net with Attention Gates and Deep Supervision."""
    
    def __init__(self, 
                 input_channels: int = 4,
                 num_classes: int = 6,
                 base_filters: int = 32,
                 use_attention: bool = True,
                 use_deep_supervision: bool = True):
        """
        Args:
            input_channels: Number of input channels (imaging + morphogen)
            num_classes: Number of segmentation classes
            base_filters: Base number of filters
            use_attention: Whether to use attention gates
            use_deep_supervision: Whether to use deep supervision
        """
        super(EnhancedUNet3D, self).__init__()
        
        self.use_attention = use_attention
        self.use_deep_supervision = use_deep_supervision
        
        # Encoder path with increased capacity
        self.enc1 = self._conv_block(input_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck with dropout for regularization
        self.bottleneck = nn.Sequential(
            self._conv_block(base_filters * 8, base_filters * 16),
            nn.Dropout3d(0.3)
        )
        
        # Decoder path with attention gates
        self.upconv4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, stride=2)
        if use_attention:
            self.att4 = AttentionGate3D(F_g=base_filters * 8, F_l=base_filters * 8, 
                                        F_int=base_filters * 4)
        self.dec4 = self._conv_block(base_filters * 16, base_filters * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
        if use_attention:
            self.att3 = AttentionGate3D(F_g=base_filters * 4, F_l=base_filters * 4, 
                                        F_int=base_filters * 2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
        if use_attention:
            self.att2 = AttentionGate3D(F_g=base_filters * 2, F_l=base_filters * 2, 
                                        F_int=base_filters)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
        if use_attention:
            self.att1 = AttentionGate3D(F_g=base_filters, F_l=base_filters, 
                                        F_int=base_filters // 2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)
        
        # Final output layer
        self.final = nn.Conv3d(base_filters, num_classes, kernel_size=1)
        
        # Deep supervision outputs
        if use_deep_supervision:
            self.ds4 = DeepSupervisionBlock(base_filters * 8, num_classes, 8)
            self.ds3 = DeepSupervisionBlock(base_filters * 4, num_classes, 4)
            self.ds2 = DeepSupervisionBlock(base_filters * 2, num_classes, 2)
            
        # Initialize weights
        self._init_weights()
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with BN and ReLU."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional deep supervision.
        
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            
        Returns:
            If training with deep supervision: List of outputs at different scales
            Otherwise: Final segmentation output
        """
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder path with attention
        dec4 = self.upconv4(bottleneck)
        if self.use_attention:
            enc4 = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        if self.use_attention:
            enc3 = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        if self.use_attention:
            enc2 = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        if self.use_attention:
            enc1 = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output
        output = self.final(dec1)
        
        # Deep supervision outputs during training
        if self.training and self.use_deep_supervision:
            ds4 = self.ds4(dec4)
            ds3 = self.ds3(dec3)
            ds2 = self.ds2(dec2)
            
            # Return all outputs for multi-scale loss
            return [output, ds2, ds3, ds4]
        
        return output


class DeepSupervisionLoss(nn.Module):
    """Loss function for deep supervision training."""
    
    def __init__(self, 
                 base_loss_fn,
                 weights: List[float] = [1.0, 0.5, 0.25, 0.125]):
        """
        Args:
            base_loss_fn: Base loss function (e.g., DiceFocalLoss)
            weights: Weights for each supervision level
        """
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.weights = weights
        
    def forward(self, outputs: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted sum of losses at different scales.
        
        Args:
            outputs: List of outputs at different scales
            target: Ground truth segmentation
            
        Returns:
            Weighted combined loss
        """
        total_loss = 0.0
        
        for i, (output, weight) in enumerate(zip(outputs, self.weights)):
            loss = self.base_loss_fn(output, target)
            total_loss += weight * loss
            
        # Normalize by sum of weights
        total_loss /= sum(self.weights)
        
        return total_loss


def test_enhanced_architecture():
    """Test the enhanced U-Net architecture."""
    
    print("🏗️ TESTING ENHANCED U-NET ARCHITECTURE")
    print("=" * 50)
    
    # Create model
    model = EnhancedUNet3D(
        input_channels=4,
        num_classes=6,
        base_filters=32,
        use_attention=True,
        use_deep_supervision=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 64, 64, 64)
    
    # Training mode (with deep supervision)
    model.train()
    outputs = model(x)
    
    if isinstance(outputs, list):
        print(f"✅ Deep supervision outputs: {len(outputs)} scales")
        for i, out in enumerate(outputs):
            print(f"   Scale {i}: {out.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"✅ Inference output: {output.shape}")
    
    # Test ONNX compatibility
    print("\n📦 Testing ONNX compatibility...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            torch.onnx.export(
                model,
                x,
                tmp.name,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=11
            )
        print("✅ ONNX export compatible!")
    except Exception as e:
        print(f"⚠️  ONNX export issue: {e}")
    
    print("\n✅ Enhanced architecture ready for training!")
    return model


if __name__ == "__main__":
    test_enhanced_architecture()
