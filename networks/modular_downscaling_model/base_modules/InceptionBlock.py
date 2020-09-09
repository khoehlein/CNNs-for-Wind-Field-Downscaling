import torch
import torch.nn as nn
from .ConvBlock import ConvBlock
from .MaxPoolBlock import MaxPoolBlock
from .ParametricModule import ParametricModule


class InceptionBlock(ParametricModule):

    __options__ = dict(
        input_channels=None,
        feature_channels=None,
        output_channels=None,
        padding_mode='replication',
        leaky_slope=0.2,
        dropout_rate=0.1,
        use_batch_norm=True
    )

    def __init__(self, **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)
        self._require_not_none('input_channels', 'output_channels')
        # https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
        # compute the number of channels for each conv layer --> number of channels should sum up
        # to num_output_channels after concatenation
        output_channels_per_block = self.output_channels // 4
        block_1x1 = ConvBlock(
            input_channels=self.input_channels,
            output_channels=output_channels_per_block,
            kernel_size=1,
            leaky_slope=self.leaky_slope,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
        )
        block_max_pool = nn.Sequential(
            ConvBlock(
                input_channels=self.input_channels,
                output_channels=output_channels_per_block,
                kernel_size=1,
                leaky_slope=self.leaky_slope,
                dropout_rate=0.,
                use_batch_norm=self.use_batch_norm
            ),
            MaxPoolBlock(
                input_channels=output_channels_per_block,
                kernel_size=3,
                padding_mode=padding_mode,
                leaky_slope=leaky_slope,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            ),
        )
        output_channels = output_channels - 2 * output_channels_per_block
        output_channels_per_block = output_channels // 2
        block_3x3 = nn.Sequential(
            ConvBlock(
                input_channels,
                feature_channels,
                kernel_size=1,
                leaky_slope=leaky_slope,
                dropout_rate=0.,
                use_batch_norm=use_batch_norm
            ),
            ConvBlock(
                feature_channels,
                output_channels_per_block,
                kernel_size=3,
                padding_mode=padding_mode,
                leaky_slope=leaky_slope,
                dropout_rate=0.,
                use_batch_norm=use_batch_norm
            ),
        )
        block_5x5 = nn.Sequential(
            ConvBlock(
                input_channels,
                feature_channels,
                kernel_size=1,
                leaky_slope=leaky_slope,
                dropout_rate=0.,
                use_batch_norm=use_batch_norm
            ),
            ConvBlock(
                feature_channels,
                output_channels - output_channels_per_block,
                kernel_size=5,
                padding_mode=padding_mode,
                leaky_slope=leaky_slope,
                dropout_rate=0.,
                use_batch_norm=use_batch_norm
            ),
        )
        self.model = nn.ModuleList([block_1x1, block_3x3, block_5x5, block_max_pool])

    def forward(self, x):
        outputs = []
        for module in self.model:
            outputs += [module(x)]
        output = torch.cat(outputs, dim=1)
        return output













