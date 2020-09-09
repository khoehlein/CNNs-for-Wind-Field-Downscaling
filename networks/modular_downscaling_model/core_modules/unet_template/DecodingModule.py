import torch.nn as nn


class DecodingModule(nn.Module):
    def __init__(
            self,
            upsampling_module: nn.Module, processing_module: nn.Module,
            inner_channels, outer_channels
    ):
        super(DecodingModule, self).__init__()
        self.upsampling_module = upsampling_module
        self.processing_module = processing_module
        self.inner_channels = inner_channels
        self.outer_channels = outer_channels

    def forward(self, inner_input, outer_input):
        output = self.upsampling_module(inner_input, outer_input)
        output = self.processing_module(output)
        return output
