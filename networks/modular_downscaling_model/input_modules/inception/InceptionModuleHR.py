import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ParametricModule, ConvBlock, InceptionBlock


class InceptionModuleHR(ParametricModule):

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
        'num_inception_blocks_hr': 1,
        'num_inception_blocks_mr': 0,
        'num_inception_blocks_lr': 1,
    }

    def __init__(self, **kwargs):
        super(InceptionModuleHR, self).__init__(**kwargs)
        if self.feature_channels_lr is None:
            self.feature_channels_lr = self.output_channels
        if self.feature_channels_mr is None:
            self.feature_channels_mr = self.feature_channels_lr
        if self.feature_channels_hr is None:
            self.feature_channels_hr = self.feature_channels_mr
        self._build_model()

    def _build_hr_module(self):
        if self.num_inception_blocks_hr > 0:
            layers = [
                ConvBlock(
                    input_channels=self.input_channels,
                    output_channels=self.feature_channels_hr,
                    kernel_size=self.kernel_size_hr,
                    padding_mode=self.padding_mode,
                    leaky_slope=self.leaky_slope,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=self.batch_norm,
                )
            ]
            for _ in range(self.num_inception_blocks_hr):
                layers += [
                    InceptionBlock(
                        input_channels=self.feature_channels_hr,
                        feature_channels=self.feature_channels_hr,
                        output_channels=self.feature_channels_hr,
                        padding_mode=self.padding_mode,
                        leaky_slope=self.leaky_slope,
                        dropout_rate=self.dropout_rate,
                        use_batch_norm=self.batch_norm,
                    )
                ]
            self.module_hr = nn.Sequential(*layers)
        else:
            self.module_hr = ConvBlock(
                input_channels=self.input_channels,
                output_channels=self.feature_channels_hr,
                kernel_size=self.kernel_size_hr,
                padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.batch_norm,
            )

    def _build_mr_module(self):
        if self.num_inception_blocks_mr > 0:
            layers = []
            for _ in range(self.num_inception_blocks_hr):
                layers += [
                    InceptionBlock(
                        input_channels=self.feature_channels_mr,
                        feature_channels=self.feature_channels_mr,
                        output_channels=self.feature_channels_mr,
                        padding_mode=self.padding_mode,
                        leaky_slope=self.leaky_slope,
                        dropout_rate=self.dropout_rate,
                        use_batch_norm=self.batch_norm,
                    )
                ]
            self.module_mr = nn.Sequential(*layers)
        else:
            self.module_mr = None

    def _build_lr_module(self):
        if self.num_inception_blocks_lr > 0:
            layers = []
            for _ in range(self.num_inception_blocks_lr):
                layers += [
                    InceptionBlock(
                        input_channels=self.feature_channels_lr,
                        feature_channels=self.feature_channels_lr,
                        output_channels=self.feature_channels_lr,
                        padding_mode=self.padding_mode,
                        leaky_slope=self.leaky_slope,
                        dropout_rate=self.dropout_rate,
                        use_batch_norm=self.batch_norm,
                    )
                ]
            layers += [
                ConvBlock(
                    input_channels=self.feature_channels_lr,
                    output_channels=self.output_channels,
                    kernel_size=self.kernel_size_lr,
                    padding_mode=self.padding_mode,
                    leaky_slope=self.leaky_slope,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=self.batch_norm,
                )
            ]
            self.module_lr = nn.Sequential(*layers)
        else:
            self.module_lr = ConvBlock(
                input_channels=self.feature_channels_lr,
                output_channels=self.output_channels,
                kernel_size=self.kernel_size_lr,
                padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.batch_norm,
            )

    def _build_model(self):
        self._build_hr_module()
        self.downsampling_23 = ConvBlock(
            input_channels=self.feature_channels_hr,
            output_channels=self.feature_channels_mr,
            kernel_size=self.kernel_size_hr,
            stride=(2, 3),
            padding_mode=self.padding_mode,
            use_batch_norm=self.batch_norm,
            leaky_slope=self.leaky_slope,
            dropout_rate=self.dropout_rate
        )
        self._build_mr_module()
        self.downsampling_21 = ConvBlock(
            input_channels=self.feature_channels_mr,
            output_channels=self.feature_channels_lr,
            kernel_size=self.kernel_size_mr,
            stride=(2, 1),
            padding_mode=self.padding_mode,
            use_batch_norm=self.batch_norm,
            leaky_slope=self.leaky_slope,
            dropout_rate=self.dropout_rate
        )
        self._build_lr_module()

    def forward(self, input_hr):
        output = self.module_hr(input_hr)
        output = self.downsampling_23(output)
        if self.module_mr is not None:
            output = self.module_mr(output)
        output = self.downsampling_21(output)
        output = self.module_lr(output)
        return output


if __name__ == '__main__':
    model = InceptionModuleHR()
