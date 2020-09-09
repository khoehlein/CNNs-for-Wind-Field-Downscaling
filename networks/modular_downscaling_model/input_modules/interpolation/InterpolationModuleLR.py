import torch
import torch.nn.functional as F
from networks.modular_downscaling_model.base_modules import ParametricModule


class InterpolationModuleLR(ParametricModule):

    __options__ = {
        'input_channels': None,
        'output_channels': None,
        'scale_factor': (4, 3),
        'interpolation_method': 'bilinear',
        'target_channels_in_lr': None
    }

    def __init__(self, **kwargs):
        super(InterpolationModuleLR, self).__init__(**kwargs)
        self._require_not_none('input_channels')
        self.output_channels = (
            len(self.target_channels_in_lr) if self.target_channels_in_lr is not None else self.input_channels
        )

    def forward(
            self, input_lr,
            scalings_lr=None, offset_lr=None,
            scalings_hr=None, offset_hr=None
    ):
        assert input_lr.size(1) == self.input_channels
        output = input_lr
        if scalings_lr is not None:
            assert self.target_channels_in_lr is not None
            output = self._use_scalings_lr(output, scalings_lr, offset_lr)
        if self.target_channels_in_lr is not None:
            output = output[:, self.target_channels_in_lr, :, :]
        output = F.interpolate(output, scale_factor=self.scale_factor, mode=self.interpolation_method)
        if scalings_hr is not None:
            output = self._use_scalings_hr(output, scalings_hr, offset_hr)
        return output

    @staticmethod
    def _use_scalings_lr(x, scalings, offset):
        output = x
        if len(scalings) > 0:
            output = list(torch.split(output, [len(s.grids) for s in scalings], dim=1))
            for i, s in enumerate(scalings):
                if s.scaler is not None:
                    output[i] = s.scaler.transform_back(output[i], offset=offset)
            output = torch.cat(output, dim=1)
        return output

    @staticmethod
    def _use_scalings_hr(x, scalings, offset):
        output = x
        if len(scalings) > 0:
            output = list(torch.split(output, [len(s.grids) for s in scalings], dim=1))
            for i, s in enumerate(scalings):
                if s.scaler is not None:
                    output[i] = s.scaler.transform(output[i], offset=offset)
            output = torch.cat(output, dim=1)
        return output