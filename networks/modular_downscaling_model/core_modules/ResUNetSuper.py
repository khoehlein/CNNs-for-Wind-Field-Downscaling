import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ConvBlock, ResNetMultiBlock, ConvMultiBlock
from networks.modular_downscaling_model.core_modules.unet_template import BaseUNet
from networks.modular_downscaling_model.core_modules.unet_template.SkipConnectionModule import SkipConnectionModule
from networks.modular_downscaling_model.core_modules.unet_template.DecodingModule import DecodingModule


class ResUNetSuper(BaseUNet):
    """
    This is a U-Net with Residual Blocks that directly performs upsampling on low-res input.
    The difference to the "old" (legacy) Residual U-Net is that after processing the input in low-res
    resolution we directly upsampling to the target resolution and perform encoding and decoding to the
    final image. This works better than the usual procedure (inserting a U-Net inbetween input processing
    and supersampling blocks). Dropout in both encode and decode parts improve overfitting
    (control with use_dropout_decode). However, one can use dropout in the encoding layers only.
    Note that we operate on low-resolution inputs as this reduced the size of the network. Operating on
    high-res is also possible (disable init_upsampling), however this increases the network size and the
    time to complete the training, and it does not lead to better results. There is also an option
    to add an additional convolution layer block to process the (upsampled) input (with feature_channels).
    """

    __options__ = {
        'input_channels': 4,
        'kernel_size': 3,
        'padding_mode': 'replication',
        'leaky_slope': 0.2,
        'dropout_rate': 0.1,
        'batch_norm': True,
        'interpolation_mode': 'bilinear',
        'residual_blocks_per_module': 1,
        'layers_per_residual_block': 2,
        'use_dropout_decode': True,
        'init_upsampling': True,
        'init_convolution': False,
        'feature_channels': 64,
    }

    def __init__(self, **kwargs):
        super(ResUNetSuper, self).__init__(**kwargs)
        self._build_unet()
        self.output_channels = self.model.outer_channels

    def _processing_module(
            self,
            input_channels, **kwargs
    ):
        module = ResNetMultiBlock(
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

        # block = [nn.Upsample(scale_factor=scale_factor, mode=self.interpolation_mode)]

        # block += [ConvBlock(
        #     input_channels=input_channels, output_channels=output_channels,
        #     kernel_size=self.kernel_size,
        #     padding_mode=self.padding_mode,
        #     leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
        #     use_batch_norm=self.batch_norm
        # )]

        # module = nn.Sequential(*block)
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

    def _processing_module_decode(
            self,
            input_channels, **kwargs
    ):
        module = ResNetMultiBlock(
            input_channels=input_channels, feature_channels=input_channels, kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            leaky_slope=self.leaky_slope, dropout_rate=0.0,
            use_batch_norm=self.batch_norm,
            num_resnet_blocks=self.residual_blocks_per_module,
            num_hidden_layers=self.layers_per_residual_block,
        )
        return module

    def _decoding_module(
            self,
            input_channels, output_channels, scale_factor, **kwargs
    ):
        if self.use_dropout_decode:
            module = DecodingModule(
                self._upsampling_module(input_channels, output_channels, scale_factor, **kwargs),
                self._processing_module(output_channels, **kwargs),
                inner_channels=input_channels,
                outer_channels=output_channels
            )
        else:
            module = DecodingModule(
                self._upsampling_module(input_channels, output_channels, scale_factor, **kwargs),
                self._processing_module_decode(output_channels, **kwargs),
                inner_channels=input_channels,
                outer_channels=output_channels
            )
        return module

    def _build_init_module(self, input_channels, output_channels):
        """
        Build an input module to process the data. For now, we assume that input has been processed
        beforehand and we just deal with this processed input. No raw data processing is supported by default.
        However, enabling init convolution enables the network to cope with raw data, as well.
        Then, see that the input_channels match the number of input features.
        """
        block = []
        if self.init_upsampling:
            # upsample image to final target resolution
             block += [nn.Upsample(scale_factor=(4, 3), mode=self.interpolation_mode)]

        if self.init_convolution:
            # process the input with a convolution block
            block += ConvBlock(
                input_channels=input_channels, output_channels=output_channels,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                leaky_slope=self.leaky_slope, dropout_rate=self.dropout_rate,
                use_batch_norm=self.batch_norm
            )

        self.upsample_input = nn.Sequential(*block)

    def _build_unet(self):
        """
        This function overrides the BaseUNet function to build a U-Net. Note that we do not perform
        a "default" standard U-Net but included 3 additional encoding / decoding layers to reduce
        the target high-res resolution to small dimension and vice versa.
        """
        # handle either processed input or raw input data
        init_output_channels = self.feature_channels if self.init_convolution else self.input_channels
        self._build_init_module(self.input_channels, init_output_channels)

        model = self._skip_connection_module(
            6 * init_output_channels, 7 * init_output_channels,
            (2, 2),
            inner_module=None
        )
        model = self._skip_connection_module(
            5 * init_output_channels, 6 * init_output_channels,
            (2, 2),
            inner_module=model
        )
        model = self._skip_connection_module(
            4 * init_output_channels, 5 * init_output_channels,
            (3, 3),
            inner_module=model
        )

        model = self._skip_connection_module(
            3 * init_output_channels, 4 * init_output_channels,
            (2, 1),
            inner_module=model
        )

        model = self._skip_connection_module(
            2 * init_output_channels, 3 * init_output_channels,
            (1, 3),
            inner_module=model
        )

        model = self._skip_connection_module(
            init_output_channels, 2 * init_output_channels,
            (2, 1),
            inner_module=model
        )

        self.model = model

    def forward(self, features):
        """
        The difference to the standard forward method is that we first upsample the image
        to the target resolution
        """
        output = self.upsample_input(features)
        output = self.model(output)
        return output


if __name__ == '__main__':
    model = ResUNetSuper()
    print(model)
