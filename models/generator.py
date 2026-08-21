import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D Spatial Feature Maps [B, C, H, W]"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x = self.norm(x)
        return x.permute(0, 3, 1, 2) # [B, C, H, W]

class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (Restormer)
    Computes cross-covariance across channels O(C^2) instead of spatial pixels O(N^2),
    enabling efficient memory usage and scale-invariant attention.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (Restormer)
    Controls feature flow with depthwise convolutions and GELU gating.
    """
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class RestormerBlock(nn.Module):
    """Restormer Transformer Block combining MDTA and GDFN with LayerNorm."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SEMSwinIRGenerator(nn.Module):
    """
    Restormer-inspired Super-Resolution Generator for SEM Images.
    Optimized for high-resolution micro-texture super-resolution with O(C^2) memory footprint.
    """
    def __init__(self, in_channels=1, out_channels=1, embed_dim=48, num_blocks=4, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # 2. Deep Transformer Feature Extraction
        self.transformer_blocks = nn.ModuleList([
            RestormerBlock(embed_dim, num_heads=4) for _ in range(num_blocks)
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # 3. High-Resolution Sub-Pixel Upsampling
        upsample_layers = []
        if scale_factor == 2:
            upsample_layers.extend([
                nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ])
        elif scale_factor == 4:
            for _ in range(2):
                upsample_layers.extend([
                    nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU()
                ])
        elif scale_factor == 8:
            for _ in range(3):
                upsample_layers.extend([
                    nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU()
                ])
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}. Choose 2, 4, or 8.")

        self.upsample = nn.Sequential(*upsample_layers)

        # 4. Reconstruction Layer
        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x_first = self.conv_first(x)

        feat = x_first
        for block in self.transformer_blocks:
            feat = block(feat)
        feat = self.conv_after_body(feat)
        feat = feat + x_first

        up = self.upsample(feat)
        out = self.conv_last(up)
        return out

if __name__ == "__main__":
    netG = SEMSwinIRGenerator(scale_factor=2)
    x = torch.randn(2, 1, 128, 128)
    out = netG(x)
    print(f"Restormer Generator Input: {x.shape} -> Output: {out.shape}")
