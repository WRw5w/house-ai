import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p_drop: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if p_drop > 0:
            layers.append(nn.Dropout2d(p_drop))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p_drop: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p_drop: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, p_drop)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                   diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_features: int = 64, p_drop: float = 0.0):
        super().__init__()
        f = base_features

        self.enc1 = DoubleConv(in_channels, f, p_drop)
        self.enc2 = Down(f, f * 2, p_drop)
        self.enc3 = Down(f * 2, f * 4, p_drop)
        self.enc4 = Down(f * 4, f * 8, p_drop)

        self.bottleneck = Down(f * 8, f * 16, p_drop)

        self.dec4 = Up(f * 16, f * 8, p_drop)
        self.dec3 = Up(f * 8, f * 4, p_drop)
        self.dec2 = Up(f * 4, f * 2, p_drop)
        self.dec1 = Up(f * 2, f, p_drop)

        self.out_conv = nn.Conv2d(f, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Bottleneck
        bn = self.bottleneck(x4)

        # Decoder
        d4 = self.dec4(bn, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        # Output logic for dip/s2s: We don't predict residual like N2V.
        # In DIP, network outputs image directly: output = model(z).
        # In Self2Self, network outputs image directly: output = model(masked_y).
        # Standard U-Net residual output = x - self.out_conv(d1).
        # But we need direct output. Let's make an option or just output directly.
        # Wait, the original model.py in exp10v2 predicts residual: return x - self.out_conv(d1)
        # We can pass a flag or just remove the residual connection if it's S2S/DIP.
        # Actually, if we output residual `return x - self.out_conv(d1)`,
        # for DIP input is random noise `z`, so `return z - out_conv`. That doesn't make sense as it forces the output to be close to `z`.
        # So we MUST output directly.
        # Let's add a parameter `residual=False`.
        return self.out_conv(d1)
