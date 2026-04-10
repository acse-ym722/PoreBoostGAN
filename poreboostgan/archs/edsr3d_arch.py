from torch import nn as nn

from poreboostgan.utils.registry import ARCH_REGISTRY


class PixelShuffle3D(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = int(scale)
        if self.scale <= 1:
            raise ValueError(f'PixelShuffle3D scale must be > 1, got {scale}.')

    def forward(self, x):
        b, c, d, h, w = x.shape
        r = self.scale
        out_c = c // (r**3)
        if out_c * (r**3) != c:
            raise ValueError(f'Channel size {c} is not divisible by scale^3={r**3}.')
        x = x.view(b, out_c, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(b, out_c, d * r, h * r, w * r)


class ResidualBlockNoBN3D(nn.Module):
    """Residual block without BatchNorm, used by EDSR3D."""

    def __init__(self, num_feat=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * self.res_scale


class UpsampleBlock3D(nn.Module):
    """PixelShuffle3D upsampling block for x2/x3/x4/x8 scales."""

    def __init__(self, scale, num_feat):
        super().__init__()
        if scale not in (2, 3, 4, 8):
            raise ValueError(f'Unsupported scale: {scale}. Supported scales: x2/x3/x4/x8.')

        layers = []
        if scale in (2, 4, 8):
            current_scale = scale
            while current_scale > 1:
                layers.append(nn.Conv3d(num_feat, num_feat * 8, 3, 1, 1))
                layers.append(PixelShuffle3D(2))
                current_scale //= 2
        else:
            layers.append(nn.Conv3d(num_feat, num_feat * 27, 3, 1, 1))
            layers.append(PixelShuffle3D(3))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class EDSR3D(nn.Module):
    """3D EDSR network for volumetric super-resolution."""

    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=0.1):
        super().__init__()
        self.head = nn.Conv3d(num_in_ch, num_feat, 3, 1, 1)
        body = [ResidualBlockNoBN3D(num_feat=num_feat, res_scale=res_scale) for _ in range(num_block)]
        body.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsample = UpsampleBlock3D(scale=upscale, num_feat=num_feat)
        self.tail = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body(feat)
        feat = feat + res
        out = self.tail(self.upsample(feat))
        return out
