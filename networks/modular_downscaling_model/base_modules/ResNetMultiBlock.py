import torch.nn as nn

from .ResNetBlock import ResNetBlock
from .ParametricModule import ParametricModule


class ResNetMultiBlock(ParametricModule):

    __options__ = dict(
        input_channels=None,
        feature_channels=None,
        output_channels=None,
        kernel_size=5, padding_mode='replication',
        leaky_slope=0.2, dropout_rate=0.0,
        use_batch_norm=True,
        num_resnet_blocks=1, num_hidden_layers=1,
        output_activation=True,
    )

    def __init__(self, **kwargs):
        if 'output_channels' in kwargs:
            assert kwargs['output_channels'] is None or kwargs['output_channels'] == kwargs['input_channels']
        kwargs.update({'output_channels': kwargs['input_channels']})
        super(ResNetMultiBlock, self).__init__(**kwargs)
        blocks = []
        for i in range(self.num_resnet_blocks):
            blocks += [
                ResNetBlock(
                    input_channels=self.input_channels,
                    feature_channels=self.feature_channels,
                    kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode,
                    leaky_slope=self.leaky_slope,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=self.use_batch_norm,
                    num_hidden_layers=self.num_hidden_layers,
                    output_activation=True if (i < self.num_resnet_blocks - 1) else self.output_activation
                ),
            ]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        output = self.model(x)
        return output
