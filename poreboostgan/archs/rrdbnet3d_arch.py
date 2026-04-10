import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from poreboostgan.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer


class ResidualDenseBlock3D(nn.Module):
    """3D Residual Dense Block used in RRDBNet3D."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv3d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv3d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv3d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv3d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB3D(nn.Module):
    """3D Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB3D, self).__init__()
        self.rdb1 = ResidualDenseBlock3D(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock3D(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock3D(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet3D(nn.Module):
    """RRDBNet for volumetric (3D) super-resolution."""

    def __init__(self, num_in_ch, num_out_ch, scale=2, num_feat=64, num_block=12, num_grow_ch=32):
        super(RRDBNet3D, self).__init__()
        if scale not in (1, 2, 4, 8):
            raise ValueError(f'Unsupported scale={scale}. Only 1/2/4/8 are supported.')
        self.scale = scale

        self.conv_first = nn.Conv3d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB3D, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv3d(num_feat, num_feat, 3, 1, 1)

        self.conv_up = nn.ModuleList()
        if scale > 1:
            num_up = int(math.log2(scale))
            for _ in range(num_up):
                self.conv_up.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))

        self.conv_hr = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, return_feats=False):
        feat = self.conv_first(x)

        trunk = feat
        for block in self.body:
            trunk = block(trunk)
        last_feat = trunk
        feat = feat + self.conv_body(trunk)

        for conv_up in self.conv_up:
            feat = F.interpolate(feat, scale_factor=2, mode='trilinear', align_corners=False)
            feat = self.lrelu(conv_up(feat))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        if return_feats:
            return out, last_feat
        return out
