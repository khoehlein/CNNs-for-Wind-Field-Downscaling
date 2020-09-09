import torch.nn as nn
from networks.modular_downscaling_model.base_modules import ParametricModule
from networks.modular_downscaling_model.core_modules.unet_template.EncodingModule import EncodingModule
from networks.modular_downscaling_model.core_modules.unet_template.UpsamplingModule import UpsamplingModule
from networks.modular_downscaling_model.core_modules.unet_template.DecodingModule import DecodingModule
from networks.modular_downscaling_model.core_modules.unet_template.SkipConnectionModule import SkipConnectionModule


class BaseUNet(ParametricModule):

    __options__ = {
        'input_channels': 64,
    }

    def __init__(self, **kwargs):
        super(BaseUNet, self).__init__(**kwargs)

    def _encoding_module(
            self,
            input_channels, output_channels, scale_factor, **kwargs
    ):
        module = EncodingModule(
            self._downsampling_module(input_channels, output_channels, scale_factor, **kwargs),
            self._processing_module(output_channels, **kwargs),
            inner_channels=output_channels,
            outer_channels=input_channels,
        )
        return module

    def _decoding_module(
            self,
            input_channels, output_channels, scale_factor, **kwargs
    ):
        module = DecodingModule(
            self._upsampling_module(input_channels, output_channels, scale_factor, **kwargs),
            self._processing_module(output_channels, **kwargs),
            inner_channels=input_channels,
            outer_channels=output_channels
        )
        return module

    def _skip_connection_module(
            self,
            outer_channels, inner_channels, scale_factor, inner_module,
            **kwargs
    ):
        module = SkipConnectionModule(
            self._encoding_module(outer_channels, inner_channels, scale_factor, **kwargs),
            self._decoding_module(inner_channels, outer_channels, scale_factor, **kwargs),
            inner_module
        )
        return module

    def _build_unet(self):
        model = self._skip_connection_module(
            6 * self.input_channels, 12 * self.input_channels,
            (2, 2),
            inner_module=None
        )
        model = self._skip_connection_module(
            3 * self.input_channels, 6 * self.input_channels,
            (2, 2),
            inner_module=model
        )
        model = self._skip_connection_module(
            self.input_channels, 3 * self.input_channels,
            (3, 3),
            inner_module=model
        )
        self.model = model

    def _upsampling_module(self, input_channels, output_channels, scale_factor, **kwargs):
        module = UpsamplingModule(
            self._upsampling_resolution_module(input_channels, output_channels, scale_factor, **kwargs),
            self._upsampling_feature_module(input_channels + output_channels, output_channels, scale_factor, **kwargs)
            # self._upsampling_feature_module(2 * output_channels, output_channels, scale_factor, **kwargs)
        )
        return module

    def _upsampling_resolution_module(self, *args, **kwargs):
        raise NotImplementedError()

    def _upsampling_feature_module(self, *args, **kwargs):
        raise NotImplementedError()

    def _downsampling_module(self, *args, **kwargs):
        raise NotImplementedError()

    def _processing_module(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, features):
        return self.model(features)
