from torch import nn as nn

from poreboostgan.utils.registry import ARCH_REGISTRY


class ResidualBlockNoBN(nn.Module):
    """Residual block without BatchNorm, used by EDSR."""

    def __init__(self, num_feat=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * self.res_scale


class UpsampleBlock(nn.Module):
    """PixelShuffle upsampling block for x2/x4 scales."""

    def __init__(self, scale, num_feat):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f'Unsupported scale: {scale}. Only x2 and x4 are supported.')

        layers = []
        current_scale = scale
        while current_scale > 1:
            layers.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            layers.append(nn.PixelShuffle(2))
            current_scale //= 2
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """Enhanced Deep Super-Resolution (EDSR) network."""

    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        body = [ResidualBlockNoBN(num_feat=num_feat, res_scale=res_scale) for _ in range(num_block)]
        body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsample = UpsampleBlock(scale=upscale, num_feat=num_feat)
        self.tail = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body(feat)
        feat = feat + res
        out = self.tail(self.upsample(feat))
        return out
