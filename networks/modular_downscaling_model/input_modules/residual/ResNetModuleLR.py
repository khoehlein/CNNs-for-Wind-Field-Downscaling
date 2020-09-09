from networks.modular_downscaling_model.base_modules import ParametricModule, ConvBlock, ResNetMultiBlock


class ResNetModuleLR(ParametricModule):

    __options__ = {
        'input_channels': 6,
        'feature_channels': None,
        'output_channels': 32,
        'kernel_size': 5,
        'padding_mode': 'replication',
        'leaky_slope': 0.1,
        'dropout_rate': 0.1,
        'batch_norm': True,
        'num_residual_blocks': 1,
        'layers_per_residual_block': 2,
    }

    def __init__(self, **kwargs):
        super(ResNetModuleLR, self).__init__(**kwargs)
        if self.feature_channels is None:
            self.feature_channels = self.output_channels
        self._build_model()

    def _build_model(self):
        self.input_module = ConvBlock(
            input_channels=self.input_channels,
            output_channels=self.feature_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            leaky_slope=self.leaky_slope,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.batch_norm,
        )
        if self.num_residual_blocks > 0:
            self.processing_module = ResNetMultiBlock(
                    input_channels=self.feature_channels,
                    feature_channels=self.feature_channels,
                    kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode,
                    leaky_slope=self.leaky_slope,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=self.batch_norm,
                    num_resnet_blocks=self.num_residual_blocks,
                    num_hidden_layers=self.layers_per_residual_block,
                )
        else:
            self.processing_module = None
        self.output_module = ConvBlock(
            input_channels=self.feature_channels,
            output_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            leaky_slope=self.leaky_slope,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.batch_norm,
        )

    def forward(self, input_lr):
        output = self.input_module(input_lr)
        if self.processing_module is not None:
            output = self.processing_module(output)
        output = self.output_module(output)
        return output


if __name__ == '__main__':
    model = ResNetModuleLR()
