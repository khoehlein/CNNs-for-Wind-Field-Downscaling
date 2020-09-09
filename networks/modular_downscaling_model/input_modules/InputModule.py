import torch
import torch.nn as nn


class InputModule(nn.Module):
    def __init__(self, module_lr, module_hr=None, module_combined=None):
        super(InputModule, self).__init__()
        output_channels = self._verify_compatibility(module_lr, module_hr, module_combined)
        self.module_lr = module_lr
        self.module_hr = module_hr
        self.module_combined = module_combined
        self.output_channels = output_channels

    @staticmethod
    def _verify_compatibility(module_lr, module_hr, module_combined):
        for m in [module_lr, module_hr, module_combined]:
            assert hasattr(m, 'output_channels') or (m is None)
        total_output_channels = module_lr.output_channels
        if module_hr is not None:
            total_output_channels += module_hr.output_channels
        if module_combined is not None:
            assert hasattr(module_combined, 'input_channels')
            assert module_combined.input_channels == total_output_channels
            total_output_channels = module_combined.output_channels
        return total_output_channels

    def forward(self, input_lr, input_hr=None):
        output = self.module_lr(input_lr)
        if self.module_hr is not None:
            assert input_hr is not None
            output = torch.cat([output, self.module_hr(input_hr)], dim=1)
        if self.module_combined is not None:
            output = self.module_combined(output)
        return output
