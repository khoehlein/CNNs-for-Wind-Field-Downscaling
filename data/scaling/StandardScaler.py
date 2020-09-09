import torch
import numpy as np

from .DataScaler import DataScaler


class StandardScaler(DataScaler):
    def __init__(
            self,
            bias=None, scale=None,
            channels=None, dtype=None, device=None,
            **kwargs
    ):
        super(StandardScaler, self).__init__(
            bias=bias, scale=scale,
            channels=channels, dtype=dtype, device=device,
            **kwargs
        )

    def compute_scalings_numpy(self, data: np.ndarray, dim=None, unbiased=None):
        if unbiased is None:
            ddof = 1
        else:
            ddof = 1 if unbiased else 0
        bias = np.mean(data, axis=dim, keepdims=True)
        scale = np.std(data, axis=dim, keepdims=True, ddof=ddof)
        return bias, scale

    def compute_scalings_torch(self, data: torch.Tensor, dim=None, unbiased=None):
        if unbiased is None:
            unbiased = True
        if dim is None:
            dim = list(range(len(data.shape)))
        bias = torch.mean(data, dim, keepdim=True)
        scale = torch.std(data, dim, keepdim=True, unbiased=unbiased)
        return bias, scale
