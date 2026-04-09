import collections.abc
import math
import warnings
from itertools import repeat

import torch
from torch import nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                init.kaiming_normal_(child.weight, **kwargs)
                child.weight.data *= scale
                if child.bias is not None:
                    child.bias.data.fill_(bias_fill)
            elif isinstance(child, nn.Linear):
                init.kaiming_normal_(child.weight, **kwargs)
                child.weight.data *= scale
                if child.bias is not None:
                    child.bias.data.fill_(bias_fill)
            elif isinstance(child, _BatchNorm):
                init.constant_(child.weight, 1)
                if child.bias is not None:
                    child.bias.data.fill_(bias_fill)


def make_layer(block, num_blocks, **kwargs):
    return nn.Sequential(*(block(**kwargs) for _ in range(num_blocks)))


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_.', stacklevel=2)

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
