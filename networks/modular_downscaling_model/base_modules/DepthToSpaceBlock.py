import numbers
import numpy as np
import torch
from .ParametricModule import ParametricModule


def depth_to_space(x, scale_factor):
    if isinstance(scale_factor, int):
        scale_factor = (scale_factor, )
    if isinstance(scale_factor, list):
        scale_factor = tuple(scale_factor)
    dim = len(scale_factor)
    shape = x.size()
    assert len(shape) > dim, \
        "[ERROR] Input tensor must have at least one dimensions more than indicated by tuple <scale_factor>."
    assert shape[-(dim + 1)] % np.prod(scale_factor) == 0, \
        "[ERROR] Number of features must be a multiple of the number of pixels per patch."
    depth_new = int(shape[-(dim + 1)] / np.prod(scale_factor))
    shape_new = shape[:-(dim+1)] + (depth_new,) + scale_factor + shape[-dim:]
    x = x.view(*shape_new)
    shape_new = []
    for dim_curr in range(dim):
        x = torch.cat(x.split(1, dim=-(dim_curr+1)), dim=-(dim+dim_curr+1))
        shape_new += [scale_factor[dim_curr] * shape[-(dim-dim_curr)]]
    shape_new = shape[:-(dim+1)] + (depth_new,) + tuple(shape_new)
    x = x.view(*shape_new)
    return x


class DepthToSpaceBlock(ParametricModule):
    __options__ = dict(
        input_channels=None,
        output_channels=None,
        scale_factor=None
    )

    def __init__(self, **kwargs):
        assert 'input_channels' in kwargs
        assert 'scale_factor' in kwargs
        super(DepthToSpaceBlock, self).__init__(**kwargs)
        self._require_not_none('input_channels', 'scale_factor')
        if isinstance(self.scale_factor, numbers.Number):
            self.scale_factor = (self.scale_factor,) * 2
        assert self.input_channels % np.prod(self.scale_factor) == 0
        self.output_channels = self.input_channels // np.prod(self.scale_factor)

    def forward(self, input_lr):
        return depth_to_space(input_lr, self.scale_factor)
