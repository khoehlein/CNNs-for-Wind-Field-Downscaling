import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ConvBlock
from networks.modular_downscaling_model.core_modules.unet_template import BaseUNet


class StandardUNet(BaseUNet):

    __options__ = {
        'input_channels': None,
        'kernel_size': 3,
        'padding_mode': 'replication',
        'leaky_slope': 0.1,
        'dropout_rate': 0.1,
        'batch_norm': True,
        'interpolation_mode': 'bilinear',
        'layers_per_module': 2,
    }

    def __init__(self, **kwargs):
        super(StandardUNet, self).__init__(**kwargs)
        self._build_unet()
        self.output_channels = self.model.outer_channels

    def _processing_module(
            self,
            input_channels, **kwargs
    ):
        if 'feature_channels' in kwargs.keys():
            feature_channels = kwargs['feature_channels']
        else:
            feature_channels = input_channels
        if self.layers_per_module == 0:
            module = nn.Identity()
        elif self.layers_per_module == 1:
            module = ConvBlock(
                input_channels=input_channels, output_channels=input_channels, kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                use_batch_norm=self.batch_norm,
                leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
            )
        else:
            layers = [
                ConvBlock(
                    input_channels=input_channels, output_channels=feature_channels, kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode,
                    use_batch_norm=self.batch_norm,
                    leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
                )
            ]
            for i in range(self.layers_per_module - 2):
                layers += [
                    ConvBlock(
                        input_channels=feature_channels, output_channels=feature_channels, kernel_size=self.kernel_size,
                        padding_mode=self.padding_mode,
                        use_batch_norm=self.batch_norm,
                        leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
                    )
                ]
            layers += [
                ConvBlock(
                    input_channels=feature_channels, output_channels=input_channels, kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode,
                    use_batch_norm=self.batch_norm,
                    leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
                )
            ]
            module = nn.Sequential(*layers)
        return module

    def _downsampling_module(
            self,
            input_channels, output_channels, scale_factor, **kwargs
    ):
        module = ConvBlock(
            input_channels=input_channels, output_channels=output_channels,
            stride=scale_factor, kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
            use_batch_norm=self.batch_norm,
        )
        return module

    def _upsampling_resolution_module(
            self,
            input_channels, output_channels, scale_factor, **kwargs
    ):
        module = nn.Upsample(scale_factor=scale_factor, mode=self.interpolation_mode)
        return module

    def _upsampling_feature_module(
            self,
            input_channels, output_channels, scale_factor, **kwargs
    ):
        module = ConvBlock(
            input_channels=input_channels, output_channels=output_channels, kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
            use_batch_norm=self.batch_norm
        )
        return module

    # @staticmethod
    # def create_parameter_pool():
    #     padding_modes = ['zero', 'replication', 'reflection']


if __name__ == '__main__':
    model = StandardUNet()
    print(model)
