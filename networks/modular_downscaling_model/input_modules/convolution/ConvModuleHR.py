from networks.modular_downscaling_model.base_modules import ParametricModule, ConvBlock, ConvMultiBlock


class ConvModuleHR(ParametricModule):

    __options__ = {
        'input_channels': 2,
        'feature_channels_hr': None,
        'feature_channels_mr': None,
        'feature_channels_lr': None,
        'output_channels': 32,
        'kernel_size_hr': 7,
        'kernel_size_mr': 5,
        'kernel_size_lr': 3,
        'padding_mode': 'replication',
        'leaky_slope': 0.1,
        'dropout_rate': 0.1,
        'batch_norm': True,
        'num_convolution_blocks_hr': 1,
        'num_convolution_blocks_mr': 0,
        'num_convolution_blocks_lr': 1,
    }

    def __init__(self, **kwargs):
        super(ConvModuleHR, self).__init__(**kwargs)
        if self.feature_channels_lr is None:
            self.feature_channels_lr = self.output_channels
        if self.feature_channels_mr is None:
            self.feature_channels_mr = self.feature_channels_lr
        if self.feature_channels_hr is None:
            self.feature_channels_hr = self.feature_channels_mr
        self._build_model()

    def _build_model(self):
        if self.num_convolution_blocks_hr > 0:
            self.convolutions_hr = ConvMultiBlock(
                input_channels=self.input_channels,
                feature_channels=self.feature_channels_hr,
                output_channels=self.feature_channels_hr,
                kernel_size=self.kernel_size_hr,
                padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                num_convolution_blocks=self.num_convolution_blocks_hr
            )
        else:
            self.convolutions_hr = None
        self.downsampling_23 = ConvBlock(
            input_channels=self.feature_channels_hr if self.num_convolution_blocks_hr > 0 else self.input_channels,
            output_channels=self.feature_channels_mr,
            kernel_size=self.kernel_size_hr,
            stride=(2, 3),
            padding_mode=self.padding_mode,
            use_batch_norm=self.batch_norm,
            leaky_slope=self.leaky_slope,
            dropout_rate=self.dropout_rate
        )
        if self.num_convolution_blocks_mr > 0:
            self.convolutions_mr = ConvMultiBlock(
                input_channels=self.feature_channels_mr,
                feature_channels=self.feature_channels_mr,
                output_channels=self.feature_channels_mr,
                kernel_size=self.kernel_size_mr,
                padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                num_convolution_blocks=self.num_convolution_blocks
            )
        else:
            self.convolutions_mr = None
        self.downsampling_21 = ConvBlock(
            input_channels=self.feature_channels_mr,
            output_channels=self.feature_channels_lr if self.num_convolution_blocks_lr > 0 else self.output_channels,
            kernel_size=self.kernel_size_mr,
            stride=(2, 1),
            padding_mode=self.padding_mode,
            use_batch_norm=self.batch_norm,
            leaky_slope=self.leaky_slope,
            dropout_rate=self.dropout_rate
        )
        if self.num_convolution_blocks_lr > 0:
            self.convolutions_lr = ConvMultiBlock(
                input_channels=self.feature_channels_lr,
                feature_channels=self.feature_channels_lr,
                output_channels=self.output_channels,
                kernel_size=self.kernel_size_lr,
                padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                num_convolution_blocks=self.num_convolution_blocks_lr
            )
        else:
            self.convolutions_lr = None

    def forward(self, input_hr):
        output = input_hr
        if self.convolutions_hr is not None:
            output = self.convolutions_hr(output)
        output = self.downsampling_23(output)
        if self.convolutions_mr is not None:
            output = self.convolutions_mr(output)
        output = self.downsampling_21(output)
        if self.convolutions_lr is not None:
            output = self.convolutions_lr(output)
        return output


if __name__ == '__main__':
    model = ConvModuleHR()
