import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from poreboostgan.utils.registry import ARCH_REGISTRY


def window_partition_3d(x, window_size):
    """Partition 3D feature maps into local windows.

    Args:
        x: (B, D, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows * B, window_size, window_size, window_size, C)
    """
    b, d, h, w, c = x.shape
    ws = window_size
    x = x.view(b, d // ws, ws, h // ws, ws, w // ws, ws, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, ws, ws, ws, c)
    return windows


def window_reverse_3d(windows, window_size, d, h, w):
    """Reverse local windows back to 3D feature maps."""
    ws = window_size
    b = int(windows.shape[0] / ((d // ws) * (h // ws) * (w // ws)))
    x = windows.view(b, d // ws, h // ws, w // ws, ws, ws, ws, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class WindowAttention3D(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}.')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: (num_windows*B, N, C)
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock3D(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=4,
                 shift_size=0,
                 mlp_ratio=2.0,
                 qkv_bias=True,
                 drop=0.0,
                 attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)

    def _calculate_mask(self, d, h, w, device):
        if self.shift_size == 0:
            return None

        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        cnt = 0

        d_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        for ds in d_slices:
            for hs in h_slices:
                for ws_ in w_slices:
                    img_mask[:, ds, hs, ws_, :] = cnt
                    cnt += 1

        mask_windows = window_partition_3d(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size**3)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        # x: (B, D, H, W, C), where D/H/W are multiples of window_size
        b, d, h, w, c = x.shape
        ws = self.window_size
        ss = self.shift_size

        shortcut = x
        x = self.norm1(x)

        if ss > 0:
            shifted_x = torch.roll(x, shifts=(-ss, -ss, -ss), dims=(1, 2, 3))
            attn_mask = self._calculate_mask(d, h, w, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition_3d(shifted_x, ws).view(-1, ws**3, c)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, ws, ws, ws, c)
        shifted_x = window_reverse_3d(attn_windows, ws, d, h, w)

        if ss > 0:
            x = torch.roll(shifted_x, shifts=(ss, ss, ss), dims=(1, 2, 3))
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


@ARCH_REGISTRY.register()
class SwinIR3D(nn.Module):
    """Lightweight SwinIR-style 3D SR network for cubic super-resolution."""

    def __init__(self,
                 upscale=4,
                 in_chans=1,
                 embed_dim=48,
                 depths=(4, 4, 4),
                 num_heads=(4, 4, 4),
                 window_size=4,
                 mlp_ratio=2.0,
                 img_range=1.0):
        super().__init__()
        if upscale not in (2, 4, 8):
            raise ValueError(f'Unsupported upscale={upscale}.')
        if len(depths) != len(num_heads):
            raise ValueError('depths and num_heads must have the same length.')

        self.upscale = upscale
        self.window_size = window_size
        self.img_range = img_range

        self.conv_first = nn.Conv3d(in_chans, embed_dim, 3, 1, 1)

        blocks = []
        for stage_depth, stage_heads in zip(depths, num_heads):
            for i in range(stage_depth):
                shift = 0 if i % 2 == 0 else window_size // 2
                blocks.append(
                    SwinTransformerBlock3D(
                        dim=embed_dim,
                        num_heads=stage_heads,
                        window_size=window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio))
        self.blocks = nn.ModuleList(blocks)

        self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)

        num_up = int(math.log2(upscale))
        self.conv_up = nn.ModuleList([nn.Conv3d(embed_dim, embed_dim, 3, 1, 1) for _ in range(num_up)])
        self.conv_hr = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv3d(embed_dim, in_chans, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _pad_to_window_size(self, x):
        # x: (B, C, D, H, W)
        _, _, d, h, w = x.shape
        ws = self.window_size
        pd = (ws - d % ws) % ws
        ph = (ws - h % ws) % ws
        pw = (ws - w % ws) % ws
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode='reflect')
        return x, (pd, ph, pw)

    def _unpad(self, x, pads):
        pd, ph, pw = pads
        if pd > 0:
            x = x[:, :, :-pd, :, :]
        if ph > 0:
            x = x[:, :, :, :-ph, :]
        if pw > 0:
            x = x[:, :, :, :, :-pw]
        return x

    def forward(self, x):
        x = x / self.img_range
        feat = self.conv_first(x)
        feat_padded, pads = self._pad_to_window_size(feat)

        x_cl = feat_padded.permute(0, 2, 3, 4, 1).contiguous()
        for blk in self.blocks:
            x_cl = blk(x_cl)
        body = x_cl.permute(0, 4, 1, 2, 3).contiguous()
        body = self._unpad(body, pads)

        feat = feat + self.conv_after_body(body)
        for conv in self.conv_up:
            feat = F.interpolate(feat, scale_factor=2, mode='trilinear', align_corners=False)
            feat = self.lrelu(conv(feat))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        out = out * self.img_range
        return out
