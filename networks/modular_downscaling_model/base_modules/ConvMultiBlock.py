import torch.nn as nn
from .ConvBlock import ConvBlock
from .ParametricModule import ParametricModule


class ConvMultiBlock(ParametricModule):

    __options__ = dict(
        input_channels=None,
        feature_channels=None,
        output_channels=None,
        kernel_size=5,
        padding_mode='replication',
        leaky_slope=0.1,
        dropout_rate=0.1,
        batch_norm=True,
        num_convolution_blocks=2,
    )

    def __init__(self, **kwargs):
        super(ConvMultiBlock, self).__init__(**kwargs)
        self._require_not_none('input_channels', 'output_channels')
        self._build_model()

    def _build_model(self):
        layers = []
        # build input convolution to produce n feature channels
        layers += [
            ConvBlock(
                input_channels = self.input_channels,
                output_channels = self.feature_channels if self.num_convolution_blocks > 1 else self.output_channels,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode, use_batch_norm=self.batch_norm,
                leaky_slope=self.leaky_slope,
                dropout_rate=self.dropout_rate if self.num_convolution_blocks == 1 else 0.0
            )
        ]
        # build conv blocks in between
        for i in range(self.num_convolution_blocks - 2):
            layers += [
                ConvBlock(
                    input_channels=self.feature_channels,
                    output_channels=self.feature_channels,
                    kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode, use_batch_norm=self.batch_norm,
                    leaky_slope=self.leaky_slope,
                    dropout_rate=0.0
                )
            ]

        # build output convolution to produce n target channels
        if self.num_convolution_blocks > 1:
            layers += [
                ConvBlock(
                    input_channels=self.feature_channels,
                    output_channels=self.output_channels,
                    kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode, use_batch_norm=self.batch_norm,
                    leaky_slope=self.leaky_slope,
                    dropout_rate=self.dropout_rate if self.num_convolution_blocks == 1 else 0.0
                )
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)