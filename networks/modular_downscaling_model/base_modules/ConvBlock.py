import numbers
import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ParametricModule


class ConvBlock(ParametricModule):

    __options__ = {
        'input_channels': None,
        'output_channels': None,
        'kernel_size': 5,
        'stride': (1, 1),  # can be used for downsampling
        'padding': 'keep_dimensions',
        'padding_mode': 'replication',
        'use_batch_norm': True,
        'leaky_slope': 0.2, 'dropout_rate': 0.0
    }

    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self._require_not_none('input_channels', 'output_channels')
        assert(isinstance(self.stride, (tuple, list)))
        if self.padding == 'keep_dimensions':
            # compute padding to keep same dimension
            if isinstance(self.kernel_size, numbers.Number):
                kernel_size_half = (self.kernel_size // 2, ) * 2
            elif isinstance(self.kernel_size, (tuple, list)):
                kernel_size_half = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
            else:
                raise Exception()
            self.padding = kernel_size_half
        else:
            assert(isinstance(self.padding, (int, tuple, list)))
        conv_block = []
        p = self.padding
        if not self.padding == 0 or self.padding == (0, 0) or self.padding == [0, 0]:
            if isinstance(self.padding, (tuple, list)) and len(self.padding) == 2:
                # handle asymmetric case
                self.padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            p = 0
            if self.padding_mode == 'reflection':
                conv_block += [nn.ReflectionPad2d(self.padding)]
            elif self.padding_mode == 'replication':
                conv_block += [nn.ReplicationPad2d(self.padding)]
            elif self.padding_mode == 'zero':
                # conv_block += [nn.ZeroPad2d(padding)]
                p = self.padding
            else:
                raise NotImplementedError('Padding mode <{}> not implemented'.format(self.padding_mode))
        conv_block += [
            nn.Conv2d(self.input_channels, self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=p)
        ]
        # Batch normalization
        if self.use_batch_norm:
            conv_block += [nn.BatchNorm2d(self.output_channels)]
        # LeakyReLU
        if isinstance(self.leaky_slope, (int, float)):
            if self.leaky_slope != 1.0:
                conv_block += [nn.LeakyReLU(float(self.leaky_slope), inplace=True)]
        elif self.leaky_slope == 'p':
            conv_block += [nn.PReLU(num_parameters=self.output_channels)]
        else:
            raise Exception()
        # Dropout layer
        if self.dropout_rate > 0.0:
            conv_block += [nn.Dropout2d(self.dropout_rate, inplace=False)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        output = self.conv_block(x)
        return output


if __name__ == '__main__':
    c = ConvBlock(input_channels=1, target_channels=2)