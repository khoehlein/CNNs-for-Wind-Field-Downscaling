import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ParametricModule, ConvBlock, ConvMultiBlock, ResamplingBlock2D


class ConvSupersamplingModule(ParametricModule):

    __options__ = {
        'input_channels': None,
        'feature_channels': None,
        'output_channels': None,
        'kernel_size': 3,
        'padding_mode': 'replication',
        'leaky_slope': 0.1,
        'dropout_rate': 0.1,
        'batch_norm': True,
        'num_convolution_blocks': 1,
        'scale_factor': (4, 3),
        'interpolation_mode': 'bilinear',
        'output_activation': True
    }

    def __init__(self, **kwargs):
        super(ConvSupersamplingModule, self).__init__(**kwargs)
        if self.feature_channels is None:
            self.feature_channels = self.output_channels
        self._build_model()

    def _build_model(self):
        self.upsampling_module = nn.Sequential(
            ResamplingBlock2D(
                input_channels=self.input_channels,
                scale_factor=self.scale_factor,
                interpolation_mode=self.interpolation_mode),
            ConvBlock(
                input_channels=self.input_channels,
                output_channels=self.feature_channels,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.batch_norm,
            ),
        )
        if self.num_convolution_blocks > 0:
            self.processing_module = ConvMultiBlock(
                input_channels=self.feature_channels,
                feature_channels=self.feature_channels,
                output_channels=self.feature_channels,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                num_convolution_blocks=self.num_convolution_blocks
            )
        else:
            self.processing_module = None
        self.output_module = ConvBlock(
            input_channels=self.feature_channels,
            output_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            leaky_slope=self.leaky_slope if self.output_activation else 1.0,
            dropout_rate=self.dropout_rate if self.output_activation else 0.0,
            use_batch_norm=self.batch_norm if self.output_activation else False,
        )

    def forward(self, input_lr):
        output = self.upsampling_module(input_lr)
        if self.processing_module is not None:
            output = self.processing_module(output)
        output = self.output_module(output)
        return output


if __name__ == '__main__':
    model = ConvSupersamplingModule()
