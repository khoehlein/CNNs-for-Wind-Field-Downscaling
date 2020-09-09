import torch.nn as nn
from .ConvBlock import ConvBlock
from .ParametricModule import ParametricModule


class ResNetBlock(ParametricModule):

    __options__ = dict(
        input_channels=None,
        feature_channels=None,
        output_channels=None,
        kernel_size=5,
        padding_mode='replication',
        leaky_slope=0.2, dropout_rate=0.0,
        use_batch_norm=True,
        num_hidden_layers=1,
        output_activation=True
    )

    def __init__(self, **kwargs):
        assert 'input_channels' in kwargs
        if 'output_channels' in kwargs:
            assert kwargs['output_channels'] is None or kwargs['output_channels'] == kwargs['input_channels']
        kwargs.update({'output_channels': kwargs['input_channels']})
        super(ResNetBlock, self).__init__(**kwargs)
        if self.feature_channels is None:
            self.feature_channels = self.input_channels
        layers = [
            ConvBlock(
                input_channels=self.input_channels,
                output_channels=self.feature_channels,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                use_batch_norm=self.use_batch_norm,
                leaky_slope=self.leaky_slope,
                dropout_rate=0.0
            )
        ]
        for i in range(self.num_hidden_layers - 1):
            layers += [
                ConvBlock(
                    input_channels=self.feature_channels,
                    output_channels=self.feature_channels,
                    kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode,
                    use_batch_norm=self.use_batch_norm,
                    leaky_slope=self.leaky_slope,
                    dropout_rate=0.0
                )
            ]
        layers += [
            ConvBlock(
                input_channels=self.feature_channels,
                output_channels=self.input_channels,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                use_batch_norm=self.use_batch_norm,
                leaky_slope=self.leaky_slope if self.output_activation else 1.0,
                dropout_rate=self.dropout_rate
            )
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = x + self.model(x)
        return output
