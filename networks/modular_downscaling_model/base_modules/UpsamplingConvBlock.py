from .ParametricModule import ParametricModule
from .ConvBlock import ConvBlock
from .ResamplingBlock2D import ResamplingBlock2D


class UpsamplingConvBlock(ParametricModule):

    __options__ = dict(
        input_channels=None,
        output_channels=None,
        kernel_size=5,
        padding_mode='replication',
        use_batch_norm=True,
        leaky_slope=0.2,
        dropout_rate=0.0,
        size=None,
        scale_factor=None,
        interpolation_mode='bilinear'
    )

    def __init__(self, **kwargs):
        super(UpsamplingConvBlock, self).__init__(**kwargs)
        self.upsampling = ResamplingBlock2D(
            input_channels=self.input_channels,
            size=self.size,
            scale_factor=self.scale_factor,
            interpolation_mode=self.interpolation_mode
        )
        self.convolution = ConvBlock(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            use_batch_norm=self.use_batch_norm,
            leaky_slope=self.leaky_slope,
            dropout_rate=self.dropout_rate,
        )

    def forward(self, x):
        features = self.upsampling(x)
        features = self.convolution(features)
        return features
