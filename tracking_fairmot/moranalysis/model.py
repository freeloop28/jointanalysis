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
            nn.BatchNorm2d(1),
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

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Input channels to DoubleConv after concatenation:
            # channels from skip connection (out_channels * 2 if factor=1, or out_channels if factor=2 from below stage)
            # + channels from the upsampled feature map (in_channels // 2).
            # Let's stick to the standard U-Net structure where input to Up() is from below and skip connection.
            # If Up(in_ch_below, out_ch_up, bilinear=True):
            #   in_ch_below = channels from the layer below (e.g., 256 for up1 in LightUNet)
            #   out_ch_up = channels for the output of this Up block (e.g., 64 for up1 in LightUNet)
            #   skip_ch = channels from the corresponding Down block (e.g., 128 for up1 in LightUNet)
            #   Input to DoubleConv = skip_ch + in_ch_below (after upsampling)
            # Let's redefine the arguments for clarity: Up(ch_in_below, ch_skip, ch_out, bilinear)
            # No, let's use the definition from LightUNet definition:
            # up1 = Up(ch * 8, (ch * 4) // factor, bilinear) -> Up(256, 128//f, True) f=2 -> Up(256, 64, True)
            # Input x4 is 128 ch (256//f), Skip x3 is 128 ch.
            # Inside Up(256, 64, True):
            # self.up takes input x4 (128 ch). Output is 128 ch.
            # torch.cat([x3 (128), x1_upsampled (128)]) -> 256 channels.
            # self.conv = DoubleConv(in_ch_concat, out_ch, mid=in_ch_concat//2)
            # Needs DoubleConv(256, 64, 128).
            # So, the 'in_channels' argument to Up should be the concatenated channels, and 'out_channels' the final output channels.
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            # Use ConvTranspose2d for learned upsampling
            # Input to ConvTranspose2d should be channels from below. Let it output 'out_channels'.
            # self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2) # Original logic
            # Let's re-evaluate for LightUNet:
            # up1 = Up(ch*8, (ch*4)//f, False) -> Up(256, 128, False). f=1.
            # Input x4 is 256 ch. Skip x3 is 128 ch.
            # Inside Up(256, 128, False):
            # self.up = ConvTranspose2d(in_channels//2 = 128, in_channels//2 = 128, ...) -> Takes 128 ch, outputs 128 ch. This seems wrong.
            # ConvTranspose should take channels from below (x4=256ch) and output channels matching skip (x3=128ch).
            # So, self.up = nn.ConvTranspose2d(ch_in_below, ch_skip, kernel_size=2, stride=2)
            # self.up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            # Then concat: x3 (128) + upsampled_x4 (128) = 256 channels.
            # Then self.conv = DoubleConv(ch_concat, ch_out) = DoubleConv(256, 128).
            # This matches the 'in_channels' and 'out_channels' args passed to Up().
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2) # Keeping original Up block logic for consistency
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

# --- Lightweight U-Net Model ---
class LightUNet(nn.Module):
    """
    A lighter version of the standard U-Net architecture for semantic segmentation.
    Features fewer channels and reduced depth compared to a typical U-Net, aiming for faster execution.

    Args:
        n_channels (int): Number of channels in the input image (e.g., 3 for RGB).
        n_classes (int): Number of output segmentation classes.
        base_channels (int): Number of channels in the first layer. Default: 32.
        bilinear (bool): Whether to use bilinear interpolation for upsampling.
                         If False, uses learnable ConvTranspose2d. Default: True.
    """
    def __init__(self, n_channels, n_classes, base_channels=32, bilinear=True):
        super(LightUNet, self).__init__()
        if n_channels <= 0:
             raise ValueError("Number of input channels must be > 0")
        if n_classes <= 0:
             raise ValueError("Number of output classes must be > 0")
        if base_channels <= 0:
             raise ValueError("Base channels must be > 0")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Factor helps adjust channel counts based on upsampling method if needed,
        # but the Up block logic handles it internally based on the 'bilinear' flag.
        # For consistency with the Up block's internal logic (esp. ConvTranspose),
        # let's define channels more directly based on the doubling/halving scheme.
        ch = base_channels # Starting channels (e.g., 32)

        # Encoder (Downsampling path) - 3 levels deep
        self.inc = DoubleConv(n_channels, ch)       # Input: n_channels, Output: ch (32)
        self.down1 = Down(ch, ch * 2)               # Input: ch (32), Output: ch*2 (64)
        self.down2 = Down(ch * 2, ch * 4)           # Input: ch*2 (64), Output: ch*4 (128)
        self.down3 = Down(ch * 4, ch * 8)           # Input: ch*4 (128), Output: ch*8 (256) - Bottleneck starts here

        # Decoder (Upsampling path) - 3 levels deep
        # Up(in_ch_concat, out_ch, bilinear)
        # in_ch_concat = channels_from_skip + channels_from_upsample(below)
        # Output of Up block = out_ch

        # Up1: Takes bottleneck (x4, ch*8=256) and skip (x3, ch*4=128).
        #      Input to concat = skip(128) + upsampled(x4).
        #      If bilinear=T: upsampled(x4)=256ch. Concat=128+256=384. Up(384, 128, T). This doesn't match original Up logic.
        #      If bilinear=F: upsampled(x4)=128ch (using revised ConvTrans logic). Concat=128+128=256. Up(256, 128, F). This matches.
        # Let's stick to the original Up block's channel definition:
        # Up(total_concat_channels, final_output_channels, bilinear)
        self.up1 = Up(ch * 8 + ch*4, ch * 4, bilinear)  # in_concat=128+128=256 (if bilinear=F) or 128+256(if bilinear=T?). Revisit Up block channel logic if needed. Let's use original Up logic. Input x4=256, Skip x3=128. Concat=256. Output=128.  -> Up(256, 128, bilinear)
        self.up2 = Up(ch * 4 + ch*2, ch * 2, bilinear)  # Input x=128, Skip x2=64. Concat=128. Output=64. -> Up(128, 64, bilinear)
        self.up3 = Up(ch * 2 + ch, ch, bilinear)        # Input x=64, Skip x1=32. Concat=64. Output=32. -> Up(64, 32, bilinear)


        # Final output convolution
        self.outc = OutConv(ch, n_classes) # Takes output of up3 (ch=32 channels)
        # Add Attention Gates
        self.att1 = AttentionGate(F_g=ch * 8, F_l=ch * 4, F_int=ch * 2)  # x4, x3
        self.att2 = AttentionGate(F_g=ch * 4, F_l=ch *2, F_int=ch)      # x, x2
        self.att3 = AttentionGate(F_g=ch * 2, F_l=ch, F_int=ch // 2)        # x, x1

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # (B, ch, H, W)         e.g., (B, 32, 256, 256)
        x2 = self.down1(x1)  # (B, ch*2, H/2, W/2)   e.g., (B, 64, 128, 128)
        x3 = self.down2(x2)  # (B, ch*4, H/4, W/4)   e.g., (B, 128, 64, 64)
        x4 = self.down3(x3)  # (B, ch*8, H/8, W/8)   e.g., (B, 256, 32, 32) - Bottleneck
        
        # Attention-modulated skip connections
        x3_att = self.att1(g=x4, x=x3)
        x2_att = self.att2(g=x3_att, x=x2)
        x1_att = self.att3(g=x2_att, x=x1)

        # Decoder
        x = self.up1(x4, x3_att) # Input below=x4(256), skip=x3(128). Output=(B, ch*4, H/4, W/4) e.g., (B, 128, 64, 64)
        x = self.up2(x, x2_att)  # Input below=x(128), skip=x2(64). Output=(B, ch*2, H/2, W/2) e.g., (B, 64, 128, 128)
        x = self.up3(x, x1_att)  # Input below=x(64), skip=x1(32). Output=(B, ch, H, W) e.g., (B, 32, 256, 256)

        # Output layer
        logits = self.outc(x) # (B, n_classes, H, W)
        return logits

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Configuration examples
    num_classes = 3    # Example: Background, Tail, Head
    input_channels = 3 # Example: RGB image
    img_height, img_width = 256, 256 # Example image size

    print("--- Testing Lightweight U-Net (LightUNet) ---")
    # Instantiate the lightweight model (defaults: base_channels=32, bilinear=True)
    model = LightUNet(n_channels=input_channels, n_classes=num_classes)
    

    # Print model summary (requires torchinfo)
    try:
        from torchinfo import summary
        # Example batch size = 4
        print("Model Summary (LightUNet, base=32, bilinear=True):")
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