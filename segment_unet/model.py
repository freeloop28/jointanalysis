import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate module
        F_g: number of channels in the gating signal (decoder feature)
        F_l: number of channels in the skip connection (encoder feature)
        F_int: number of intermediate channels
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='nearest')

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# --- Helper Convolutional Block ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # Using bias=False with BatchNorm is common practice
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# --- Downsampling Block ---
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# --- Upsampling Block ---
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) # Keeping original Up block logic for consistency
            self.conv = DoubleConv(in_channels, out_channels) # Input to DoubleConv is sum of channels after concat


    def forward(self, x1, x2):
        """
        x1: feature map from the upsampling path (output of the previous Up block or the bottleneck)
        x2: feature map from the corresponding downsampling path (skip connection)
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match the spatial dimensions of x2 (skip connection)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- Output Layer ---
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- U-Net Model ---
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_attention=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

        factor = 2 if bilinear else 1  # bilinear 时 bottleneck 通道减半

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)  # x5

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if self.use_attention:
            # 根据 skip/gating 通道设置 F_int = skip // 2
            ch_bottleneck = 1024 // factor
            self.att1 = AttentionGate(F_g=ch_bottleneck, F_l=512, F_int=256)
            self.att2 = AttentionGate(F_g=512 // factor, F_l=256, F_int=128)
            self.att3 = AttentionGate(F_g=256 // factor, F_l=128, F_int=64)
            self.att4 = AttentionGate(F_g=128 // factor, F_l=64,  F_int=32)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024//factor

        if self.use_attention:
            x4_skip = self.att1(g=x5, x=x4)
            x = self.up1(x5, x4_skip)            # 先上采样

            x3_skip = self.att2(g=x,  x=x3)
            x = self.up2(x, x3_skip)
            x2_skip = self.att3(g=x,  x=x2)
            x = self.up3(x, x2_skip)
            x1_skip = self.att4(g=x,  x=x1)
            x = self.up4(x, x1_skip)
        else:
            
            # x = self.up1(x5, x4_skip)
            # x = self.up2(x,  x3_skip)
            # x = self.up3(x,  x2_skip)
            # x = self.up4(x,  x1_skip)
            x = self.up1(x5, x4)
            x = self.up2(x,  x3)
            x = self.up3(x,  x2)
            x = self.up4(x,  x1)
        return self.outc(x)

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Configuration examples
    num_classes = 3    # Example: Background, Tail, Head
    input_channels = 3 # Example: RGB image
    img_height, img_width = 256, 256 # Example image size

    print("--- Testing U-Net  ---")
    # Instantiate the model (defaults: base_channels=32, bilinear=True)
    model = UNet(n_channels=input_channels, n_classes=num_classes)
    

    # Print model summary (requires torchinfo)
    try:
        from torchinfo import summary
        # Example batch size = 4
        print("Model Summary (UNet, base=64, bilinear=True):")
        # Provide input size as (batch_size, channels, height, width)
        summary(model, input_size=(4, input_channels, img_height, img_width))
    except ImportError:
        print("Install torchinfo for model summary: pip install torchinfo")
        print("\nModel Structure:")
        print(model)


    # Example forward pass
    dummy_batch = torch.randn(4, input_channels, img_height, img_width)
    print(f"\nInput shape: {dummy_batch.shape}")

    # Check model device (should be CPU by default unless moved)
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")
    dummy_batch = dummy_batch.to(model_device) # Ensure batch is on the same device

    try:
        output_logits = model(dummy_batch)
        print(f"Output logits shape: {output_logits.shape}") # Should be (Batch, Classes, Height, Width)

        # Example loss calculation (same as before)
        # Ensure target is also on the correct device
        dummy_target = torch.randint(0, num_classes, (4, img_height, img_width), device=model_device).long()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output_logits, dummy_target)
        print(f"Example CrossEntropyLoss: {loss.item():.4f}")

    except Exception as e:
        print(f"\nError during forward pass or loss calculation: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test Completed ---")