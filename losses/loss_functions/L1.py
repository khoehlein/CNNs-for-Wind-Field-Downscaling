import torch
from .LossFunction import LossFunction


class L1(LossFunction):
    def __init__(
            self,
            use_mask=True, use_scalings=False,
            batch_reduction=None, spatial_reduction=None
    ):
        super(L1, self).__init__(
            'l1',
            use_mask=use_mask, use_scalings=use_scalings,
            batch_reduction=batch_reduction, spatial_reduction=spatial_reduction
        )

    def local_deviation(self, predictions, targets, keepdim=True):
        dev = torch.abs(targets - predictions)
        dev = torch.sum(dev, dim=1, keepdim=keepdim)
        return dev