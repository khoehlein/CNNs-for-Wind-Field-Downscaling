import torch.nn.functional as F
from .ParametricModule import ParametricModule


class ResamplingBlock2D(ParametricModule):

    __options__ = dict(
        input_channels=None,
        output_channels=None,
        size=None,
        scale_factor=None,
        interpolation_mode='bilinear'
    )

    def __init__(self, **kwargs):
        if 'output_channels' in kwargs:
            assert kwargs['output_channels'] is None or kwargs['output_channels'] == kwargs['input_channels']
        kwargs.update({'output_channels': kwargs['input_channels']})
        super(ResamplingBlock2D, self).__init__(**kwargs)
        if self.size is not None:
            assert self.scale_factor is None
        if self.scale_factor is not None:
            assert self.size is None

    def forward(self, x):
        if self.size is not None:
            return F.interpolate(x, size=self.size, mode=self.interpolation_mode)

        elif self.scale_factor is not None:
            dimX = x.shape[3]
            dimY = x.shape[2]

            newSize = (dimY * self.scale_factor[0], dimX * self.scale_factor[1])
            return F.interpolate(x, size=newSize, mode=self.interpolation_mode)
        else:
            raise Exception

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.interpolation_mode
        return info