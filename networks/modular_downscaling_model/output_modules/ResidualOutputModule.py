import torch
import torch.nn as nn
import torch.nn.functional as F
from data.scaling import DataScaler


class ResidualOutputModule(nn.Module):
    def __init__(self, model, learn_residuals, scalings_lr=None, scalings_hr=None, interpolation_mode='bicubic'):
        super(ResidualOutputModule, self).__init__()
        self.learn_residuals = learn_residuals
        self._verify_scalings(scalings_lr)
        self.scalings_lr = scalings_lr
        self._verify_scalings(scalings_hr)
        self.scalings_hr = scalings_hr
        self.interpolation_mode = interpolation_mode
        self.model = model

    @staticmethod
    def _verify_scalings(scalings):
        if not (isinstance(scalings, (list, tuple)) or (scalings is None)):
            raise  AssertionError("[ERROR] Scalings mus be lists or tuples of objects of type <DataScaler> or None.")
        if scalings is not None:
            for s in scalings:
                assert isinstance(s.scaler, DataScaler)

    def forward(self, x, estimate_lr=None):
        if self.model is not None:
            output = self.model(x)
            if self.learn_residuals:
                if estimate_lr is not None:
                    if self.scalings_lr is not None:
                        estimate_lr = self._revert_scalings(estimate_lr, self.scalings_lr)
                    estimate_hr = self._interpolate(estimate_lr)
                    if self.scalings_hr is not None:
                        estimate_hr = self._apply_scalings(estimate_hr, self.scalings_hr)
                else:
                    raise NotImplementedError()
                output = output + estimate_hr
            return output
        else:
            raise AttributeError(
                '[ERROR] Child classes of <SuperResModule> must override class attribute model'
            )

    @staticmethod
    def _apply_scalings(x, scalings):
        output = x
        if scalings is not None and len(scalings) > 0:
            output = torch.split(output, [s.scaler.channels for s in scalings], dim=1)
            output = [s.scaler.transform(channel) for s, channel in zip(scalings, output)]
            output = torch.cat(output, dim=1)
        return output

    @staticmethod
    def _revert_scalings(x, scalings):
        output = x
        if scalings is not None and len(scalings) > 0:
            output = torch.split(output, [s.scaler.channels for s in scalings], dim=1)
            output = [s.scaler.transform_back(channel) for s, channel in zip(scalings, output)]
            output = torch.cat(output, dim=1)
        return output

    def _interpolate(self, estimate_lr, scale_factor=(4, 3)):
        return F.interpolate(estimate_lr, scale_factor=scale_factor, mode=self.interpolation_mode)
