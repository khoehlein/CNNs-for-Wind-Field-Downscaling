import numbers
import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ParametricModule


class SubsamplingModuleHR(ParametricModule):
    __options__ = {
        'channels': 6,
        'scale_factor': (4, 3),
        'padding_mode': 'replication'
    }

    def __init__(self, **kwargs):
        super(SubsamplingModuleHR, self).__init__(**kwargs)
        if isinstance(self.scale_factor, numbers.Number):
            self.scale_factor = [self.scale_factor] * 2
        self.input_channels = self.channels
        self.output_channels = self.channels
        # kernel_size = [2 * f - 1 for f in self.scale_factor]
        # sigma = self.scale_factor
        self._build_padding_layer(self.scale_factor)
        self.filter = nn.AvgPool2d(kernel_size=self.scale_factor, stride=self.scale_factor)

    def _build_padding_layer(self, kernel_size):
        assert isinstance(kernel_size, (list, tuple))
        padding = [k // 2 for k in kernel_size]
        if not padding == (0, 0) or padding == [0, 0]:
            if isinstance(padding, (tuple, list)) and len(padding) == 2:
                # handle asymmetric case
                padding = (padding[1], padding[1], padding[0], padding[0])
            if self.padding_mode == 'reflection':
                layer = nn.ReflectionPad2d(padding)
            elif self.padding_mode == 'replication':
                layer = nn.ReplicationPad2d(padding)
            elif self.padding_mode == 'zero':
                layer = nn.ZeroPad2d(padding)
            else:
                raise NotImplementedError('Padding mode <{}> not implemented'.format(self.padding_mode))
            self.padding_module = layer
        else:
            self.padding_module = None

    def forward(self, input_hr):
        assert input_hr.size(1) == self.channels
        output = input_hr
        if self.padding_module is not None:
            output = self.padding_module(output)
        output = self.filter(output)
        return output


if __name__ == '__main__':
    model = SubsamplingModuleHR()
