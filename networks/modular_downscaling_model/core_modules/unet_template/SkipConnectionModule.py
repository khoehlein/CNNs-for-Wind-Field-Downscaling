import torch.nn as nn
from .EncodingModule import EncodingModule
from .DecodingModule import DecodingModule


class SkipConnectionModule(nn.Module):
    def __init__(self, encoding_module, decoding_module, inner_module=None):
        super(SkipConnectionModule, self).__init__()
        self._verify_compatibility(encoding_module, decoding_module, inner_module)
        self.encoding_module = encoding_module
        self.inner_module = inner_module
        self.decoding_module = decoding_module
        self.inner_channels = encoding_module.inner_channels
        self.outer_channels = encoding_module.outer_channels

    @staticmethod
    def _verify_compatibility(encoding_module, decoding_module, inner_module):
        assert isinstance(encoding_module, EncodingModule)
        assert isinstance(decoding_module, DecodingModule)
        assert encoding_module.inner_channels == decoding_module.inner_channels
        assert encoding_module.outer_channels == decoding_module.outer_channels
        assert isinstance(inner_module, nn.Module) or (inner_module is None)
        if inner_module is not None:
            assert encoding_module.inner_channels == inner_module.outer_channels

    def forward(self, outer_features):
        inner_features = self.encoding_module(outer_features)
        if self.inner_module is not None:
            inner_features = self.inner_module(inner_features)
        outer_features = self.decoding_module(inner_features, outer_features)
        return outer_features
