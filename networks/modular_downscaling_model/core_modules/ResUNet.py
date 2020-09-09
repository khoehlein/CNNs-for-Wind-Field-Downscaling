import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ConvBlock, ResNetMultiBlock
from networks.modular_downscaling_model.core_modules.unet_template import BaseUNet


class ResUNet(BaseUNet):

    __options__ = {
        'input_channels': None,
        'kernel_size': 3,
        'padding_mode': 'replication',
        'leaky_slope': 0.1,
        'dropout_rate': 0.1,
        'batch_norm': True,
        'interpolation_mode': 'bilinear',
        'residual_blocks_per_module': 1,
        'layers_per_residual_block': 2
    }

    def __init__(self, **kwargs):
        super(ResUNet, self).__init__(**kwargs)
        self._build_unet()
        self.output_channels = self.model.outer_channels

    def _processing_module(
            self,
            input_channels, **kwargs
    ):
        module =  ResNetMultiBlock(
            input_channels=input_channels, feature_channels=input_channels, kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
            use_batch_norm=self.batch_norm,
            num_resnet_blocks=self.residual_blocks_per_module,
            num_hidden_layers=self.layers_per_residual_block,
        )
        return module

    def _downsampling_module(
            self,
            input_channels, output_channels, scale_factor, **kwargs
    ):
        module =  ConvBlock(
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
            input_channels=input_channels, output_channels=output_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
            use_batch_norm=self.batch_norm
        )
        return module

    # @staticmethod
    # def create_parameter_pool():
    #     input_channels = [64]  # 2 ** np.arange(5, 7, 1)
    #     kernel_sizes = np.arange(3, 6, 2)  # [3]  # np.arange(1, 8, 2)
    #     resblocks_per_module = [1, 2, 3]  # np.arange(1, 4, 1)
    #     layers_per_resblock = [1, 2, 3]  # np.arange(1, 4, 1)
    #     dropout_rate = np.arange(0.0, 0.5, 0.1)


if __name__ == '__main__':
    model = ResUNet()
    print(model)