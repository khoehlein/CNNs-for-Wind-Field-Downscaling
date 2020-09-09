import torch.nn as nn
from .ParametricModule import ParametricModule


class MaxPoolBlock(ParametricModule):

    __options__ = dict(
        input_channels=None,
        output_channels=None,
        kernel_size=3,
        stride=(1, 1),  # can be used for downsampling
        padding='keep_dimensions',
        padding_mode='replication',
        use_batch_norm=True,
        leaky_slope=0.2, dropout_rate=0.0

    )

    def __init__(self, **kwargs):
        if 'output_channels' in kwargs:
            assert kwargs['output_channels'] is None or kwargs['output_channels'] == kwargs['input_channels']
        kwargs.update({'output_channels': kwargs['input_channels']})
        super(MaxPoolBlock, self).__init__(**kwargs)
        self._require_not_none('input_channels')
        assert(isinstance(self.stride, (tuple, list)))
        if self.padding == 'keep_dimensions':
            # compute padding to keep same dimension
            kernel_size_half = self.kernel_size // 2
            self.padding = kernel_size_half
        else:
            assert(isinstance(self.padding, (int, tuple, list)))
        layers = []
        p = self.padding
        if not self.padding == 0 or self.padding == (0, 0) or self.padding == [0, 0]:
            if isinstance(self.padding, tuple) and len(self.padding) == 2:
                # handle asymmetric case
                self.padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            p = 0
            if self.padding_mode == 'reflection':
                layers += [nn.ReflectionPad2d(self.padding)]
            elif self.padding_mode == 'replication':
                layers += [nn.ReplicationPad2d(self.padding)]
            elif self.padding_mode == 'zero':
                # conv_block += [nn.ZeroPad2d(padding)]
                p = self.padding
            else:
                raise NotImplementedError('Padding mode <{}> not implemented'.format(self.padding_mode))
        layers += [
            nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=p)
        ]
        # Batch normalization
        if self.use_batch_norm:
            layers += [nn.BatchNorm2d(self.input_channels)]
        # LeakyReLU
        if self.leaky_slope != 1.0:
            layers += [nn.LeakyReLU(self.leaky_slope, inplace=True)]
        # Dropout layer
        if self.dropout_rate > 0.0:
            layers += [nn.Dropout2d(self.dropout_rate, inplace=False)]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_block(x)
        return output