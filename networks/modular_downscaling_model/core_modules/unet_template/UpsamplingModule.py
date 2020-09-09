import torch
import torch.nn as nn


class UpsamplingModule(nn.Module):
    def __init__(self, resolution_module, feature_module):
        super(UpsamplingModule, self).__init__()
        self.resolution_module = resolution_module
        self.feature_module = feature_module

    def forward(self, inner_input, outer_input):
        output = self.resolution_module(inner_input)
        output = torch.cat([output, outer_input], dim=1)
        output = self.feature_module(output)
        return output