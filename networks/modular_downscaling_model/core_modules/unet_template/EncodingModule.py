import torch.nn as nn


class EncodingModule(nn.Module):
    def __init__(
            self,
            downsampling_module: nn.Module, processing_module: nn.Module,
            inner_channels, outer_channels
    ):
        super(EncodingModule, self).__init__()
        self.downsampling_module = downsampling_module
        self.processing_module = processing_module
        self.inner_channels = inner_channels
        self.outer_channels = outer_channels

    def forward(self, outer_input):
        output = self.downsampling_module(outer_input)
        output = self.processing_module(output)
        return output
