import torch
from .LossFunction import LossFunction


class CosineSimilarity(LossFunction):
    def __init__(
            self,
            use_mask=True, use_scalings=False,
            batch_reduction=None, spatial_reduction=None
    ):
        super(CosineSimilarity, self).__init__(
            'cosine-similarity',
            use_mask=use_mask, use_scalings=use_scalings,
            batch_reduction=batch_reduction, spatial_reduction=spatial_reduction
        )

    def local_deviation(self, predictions, targets, keepdim=True, eps=1.e-9):
        norm = torch.sum(torch.abs(predictions) ** 2, dim=1, keepdim=keepdim) * \
               torch.sum(torch.abs(targets) ** 2, dim=1, keepdim=keepdim)
        norm = torch.sqrt(norm) + eps
        dev = torch.sum(predictions * targets, dim=1, keepdim=keepdim) / norm
        return dev