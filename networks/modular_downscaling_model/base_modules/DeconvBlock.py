import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ParametricModule


class DeconvBlock(ParametricModule):

    __options__ = {
        'input_channels': None,
        'output_channels': None,
        'kernel_size': 5,
        'stride': (1, 1),  # can be used for downsampling
        'padding': 'keep_dimensions',
        'use_batch_norm': True,
        'leaky_slope': 0.2, 'dropout_rate': 0.0
    }

    def __init__(self, **kwargs):
        super(DeconvBlock, self).__init__(**kwargs)
        self._require_not_none('input_channels', 'output_channels')
        assert(isinstance(self.stride, (tuple, list)))
        if isinstance(self.stride, int):
            self.stride = (self.stride,) * 2
        if self.padding == 'keep_dimensions':
            # compute padding to keep same dimension
            if isinstance(self.kernel_size, int):
                kernel_size_half = (self.kernel_size // 2,) * 2
            elif isinstance(self.kernel_size, (tuple, list)):
                kernel_size_half = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
            else:
                raise Exception()
            self.padding = kernel_size_half
        else:
            assert(isinstance(self.padding, (int, tuple, list)))
        self.convolution = nn.ConvTranspose2d(
            self.input_channels, self.output_channels,
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        pp_layers = []
        # Batch normalization
        if self.use_batch_norm:
            pp_layers += [nn.BatchNorm2d(self.output_channels)]
        # LeakyReLU
        if isinstance(self.leaky_slope, (int, float)):
            if self.leaky_slope != 1.0:
                pp_layers += [nn.LeakyReLU(float(self.leaky_slope), inplace=True)]
        elif self.leaky_slope == 'p':
            pp_layers += [nn.PReLU(num_parameters=self.output_channels)]
        else:
            raise Exception()
        # Dropout layer
        if self.dropout_rate > 0.0:
            pp_layers += [nn.Dropout2d(self.dropout_rate, inplace=False)]
        if len(pp_layers) > 0:
            self.post_processing = nn.Sequential(*pp_layers)
        else:
            self.post_processing = None

    def forward(self, x):
        input_size = x.shape[-2:]
        output_size = [self.stride[0] * input_size[0], self.stride[1] * input_size[1]]
        output = self.convolution(x, output_size=output_size)
        if self.post_processing is not None:
            output = self.post_processing(output)
        return output
