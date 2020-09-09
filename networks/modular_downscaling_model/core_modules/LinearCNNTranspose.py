import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.modular_downscaling_model.base_modules import ParametricModule

class LinearCNNTranspose(ParametricModule):
    __options__ = {
        'input_channels': None,
        'output_channels': None,
    }

    def __init__(self, **kwargs):
        super(LinearCNNTranspose, self).__init__(**kwargs)
        self._require_not_none('input_channels', 'output_channels')
        if isinstance(self.input_channels, (list, tuple)):
            lr_channels = self.input_channels[0]
            self.model_hr = nn.Sequential(
                nn.ReplicationPad2d((4, 4, 4, 4)),
                nn.Conv2d(self.input_channels[1], self.output_channels, 9)
            )
        else:
            lr_channels = self.input_channels
            self.model_hr = None
        self.features_lr = nn.Sequential(
            nn.ReplicationPad2d((2, 2, 2, 2)),
            nn.Conv2d(lr_channels, 5 * 5 * lr_channels, 5),
        )
        self.superres_lr = nn.ConvTranspose2d(
            5 * 5 * lr_channels, self.output_channels, (12, 9), stride=(4, 3)
        )

    def forward(self, *args):
        features = self.features_lr(args[0])
        features = self.superres_lr(features)
        features = features[:, :, 4:-4, 3:-3]
        features = features / self._get_norm(args[0])
        if self.model_hr is not None:
            features = features + self.model_hr(args[1])
        return features

    def _get_norm(self, input_lr):
        dummy = torch.ones(1, 1, *input_lr.shape[-2:], device=input_lr.device)
        return F.conv_transpose2d(dummy, torch.ones(1, 1, 12, 9, device=input_lr.device), stride=(4, 3))[:, :, 4:-4, 3:-3]


if __name__ == '__main__':
    model = LinearCNNTranspose(input_channels=[2, 2], output_channels=2)
    lr = torch.randn(1, 2, 36, 60)
    hr = torch.randn(1, 2, 144, 180)
    print(model(lr, hr).shape)
    model = LinearCNNTranspose(input_channels=2, output_channels=2)
    print(model(lr).shape)
