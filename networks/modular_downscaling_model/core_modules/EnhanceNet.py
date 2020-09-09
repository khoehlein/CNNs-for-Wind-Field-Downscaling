import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ParametricModule, ResNetMultiBlock, ResamplingBlock2D, ConvBlock


class EnhanceNet(ParametricModule):

    __options__ = {
        'input_channels': None,
        'output_channels': None,
        'feature_channels': None,
        'padding_mode': 'replication',
        'leaky_slope': 0.1,
        'dropout_rate': 0.1,
        'batch_norm': True,
        'num_residual_blocks': 10,
        'layers_per_residual_block': 2,
    }

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        if self.feature_channels is None:
            self.feature_channels = 64
        self._require_not_none('input_channels', 'output_channels')

        self.model = nn.Sequential(
            ResNetMultiBlock(
                input_channels=self.input_channels,
                feature_channels=self.input_channels,
                output_channels=self.input_channels,
                kernel_size=3, padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
                use_batch_norm=self.batch_norm,
                num_resnet_blocks=self.num_residual_blocks,
                num_hidden_layers=self.layers_per_residual_block-1,
                output_activation=True
            ),
            ResamplingBlock2D(
                input_channels=self.input_channels,
                output_channels=self.input_channels,
                size=None,
                scale_factor=(2, 1),
                interpolation_mode='nearest'
            ),
            ConvBlock(
                input_channels=self.input_channels,
                output_channels=self.input_channels,
                kernel_size=3,
                padding_mode='replication',
                use_batch_norm=True,
                leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate
            ),
            ResamplingBlock2D(
                input_channels=self.input_channels,
                output_channels=self.input_channels,
                size=None,
                scale_factor=(1, 3),
                interpolation_mode='nearest'
            ),
            ConvBlock(
                input_channels=self.input_channels,
                output_channels=self.input_channels,
                # kernel_size=[3, 5],
                kernel_size=[3, 5],
                padding_mode='replication',
                use_batch_norm=True,
                leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate
            ),
            ResamplingBlock2D(
                input_channels=self.input_channels,
                output_channels=self.input_channels,
                size=None,
                scale_factor=(2, 1),
                interpolation_mode='nearest'
            ),
            #####################################################################
            # Added by michael --> this should be exactly like the paper 1x Conv, ReLU after supersampling
            ConvBlock(
                input_channels=self.input_channels,
                output_channels=self.input_channels,
                kernel_size=3,
                padding_mode='replication',
                use_batch_norm=True,
                leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate
            ),
            # + 1x Conv, ReLU afterwards
            ConvBlock(
                input_channels=self.input_channels,
                output_channels=self.input_channels,
                kernel_size=3,
                padding_mode='replication',
                use_batch_norm=True,
                leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate
            ),
            #####################################################################
            ConvBlock(
                input_channels=self.input_channels,
                output_channels=self.output_channels,
                kernel_size=3,
                padding_mode='replication',
                use_batch_norm=False,
                leaky_slope=1.0, dropout_rate=0.0
            ),
        )

    def forward(self, features):
        return self.model(features)
