import torch
from .LossFunction import LossFunction


class MAE(LossFunction):
    def __init__(
            self,
            use_mask=True, use_scalings=False,
            batch_reduction=None, spatial_reduction=None
    ):
        super(MAE, self).__init__(
            'mae',
            use_mask=use_mask, use_scalings=use_scalings,
            batch_reduction=batch_reduction, spatial_reduction=spatial_reduction
        )

    def local_deviation(self, predictions, targets, keepdim=True):
        dev = torch.abs(targets - predictions) ** 2
        dev = torch.sqrt(torch.sum(dev, dim=1, keepdim=keepdim))
        return dev