import torch
from .LossFunction import LossFunction


class MSE(LossFunction):
    def __init__(
            self,
            use_mask=True, use_scalings=False,
            batch_reduction=None, spatial_reduction=None
    ):
        super(MSE, self).__init__(
            'mse',
            use_mask=use_mask, use_scalings=use_scalings,
            batch_reduction=batch_reduction, spatial_reduction=spatial_reduction

        )

    def local_deviation(self, predictions, targets, keepdim=True):
        dev = torch.abs(targets - predictions) ** 2
        dev = torch.sum(dev, dim=1, keepdim=keepdim)
        return dev