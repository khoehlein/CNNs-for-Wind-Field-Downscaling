import torch
import numpy as np

from .DataScaler import DataScaler


class RangeScaler(DataScaler):
    def __init__(
            self,
            bias=None, scale=None,
            channels=None, dtype=None, device=None,
            **kwargs
    ):
        super(RangeScaler, self).__init__(
            bias=bias, scale=scale,
            channels=channels, dtype=dtype, device=device,
            **kwargs
        )

    def compute_scalings_numpy(self, data: np.ndarray, dim=None, range_min=None, range_max=None):
        if range_min is None:
            range_min = -1
        if range_max is None:
            range_max = 1
        sample_min = np.amin(data, axis=dim, keepdims=True)
        sample_max = np.amax(data, axis=dim, keepdims=True)
        scale = (range_max - range_min) / (sample_max - sample_min)
        bias = (1. - scale) * sample_min
        return bias, scale

    def compute_scalings_torch(self, data: torch.Tensor, dim=None, range_min=None, range_max=None):
        if range_min is None:
            range_min = -1.
        if range_max is None:
            range_max = 1.
        if dim is None:
            dim = list(range(len(data.shape)))
        sample_min, _ = torch.min(data, dim=dim, keepdim=True)
        sample_max, _ = torch.max(data, dim=dim, keepdim=True)
        scale = (range_max - range_min) / (sample_max - sample_min)
        bias = (1. - scale) * sample_min
        return bias, scale
