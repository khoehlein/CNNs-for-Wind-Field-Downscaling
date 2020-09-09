import numbers
import numpy as np
import torch
from .ParametricModule import ParametricModule


def space_to_depth(x, scale_factor):
    if isinstance(scale_factor, int):
        scale_factor = (scale_factor, )
    dim = len(scale_factor)
    assert len(x.size()) >= dim, "[ERROR] Input tensor must have at least as many dimensions as tuple <scale_factor>."
    if len(x.size()) == dim:
        x = x.unsqueeze(0)
    for dim_curr in range(dim):
        x = torch.stack(x.split(scale_factor[dim_curr], dim=-dim), dim=-1)
    x = x.flatten(start_dim=-(2*dim+1), end_dim=-(dim+1))
    return x


class SpaceToDepthBlock(ParametricModule):

    __options__ = dict(
        input_channels=None,
        output_channels=None,
        scale_factor=None
    )

    def __init__(self, **kwargs):
        assert 'input_channels' in kwargs
        assert 'scale_factor' in kwargs
        super(SpaceToDepthBlock, self).__init__(**kwargs)
        self._require_not_none('input_channels', 'scale_factor')
        if isinstance(self.scale_factor, numbers.Number):
            self.scale_factor = (self.scale_factor,) * 2
        self.output_channels = np.prod(self.scale_factor) * self.input_channels
        
    def forward(self, input_hr):
        return space_to_depth(input_hr, self.scale_factor)