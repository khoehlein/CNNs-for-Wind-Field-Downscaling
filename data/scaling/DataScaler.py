import numpy as np
import torch
import torch.nn as nn


class DataScaler(nn.Module):
    def __init__(
            self,
            bias=None, scale=None,
            channels=None, dtype=None, device=None,
            **kwargs
    ):
        super(DataScaler, self).__init__()
        if bias is None:
            assert scale is None
            assert channels is not None
            bias = torch.zeros(1, channels, 1, 1, dtype=dtype, device=device)
            scale = torch.ones(1, channels, 1, 1, dtype=dtype, device=device)
        else:
            assert isinstance(bias, torch.Tensor)
            assert isinstance(scale, torch.Tensor)
            assert bias.size() == scale.size()
            assert bias.size(1) == channels or channels is None
            assert bias.dtype == scale.dtype
            assert bias.device == scale.device
            channels = bias.size(1)
        self.channels = channels
        self.register_buffer('bias', bias)
        self.bias.requires_grad = False
        self.register_buffer('scale', scale)
        self.scale.requires_grad = False
        self.kwargs = kwargs
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @classmethod
    def from_data(cls, data, **kwargs):
        assert len(data.shape) == 4
        channels = data.shape[1]
        if isinstance(data, torch.Tensor):
            dtype = data.dtype,
            device = data.device
        elif isinstance(data, np.ndarray):
            dtype = torch.double
            device = torch.device('cpu')
        else:
            raise Exception('[ERROR] Unknown data type.')
        scaler = cls(channels=channels, dtype=dtype, device=device, **kwargs)
        scaler.fit(data)
        return scaler

    @classmethod
    def from_parameters(cls, bias, scale, **kwargs):
        scaler = cls(bias=bias, scale=scale, **kwargs)
        return scaler

    def fit(self, data, **kwargs):
        assert isinstance(data, (np.ndarray, torch.Tensor))
        if len(kwargs) > 0:
            self.kwargs.update(kwargs)
        if isinstance(data, np.ndarray):
            self.cpu()
            bias, scale = self.compute_scalings_numpy(data, **self.kwargs)
            self.bias = torch.tensor(bias)
            self.scale = torch.tensor(scale)
        else:
            self.to(data.device)
            bias, scale = self.compute_scalings_torch(data, **self.kwargs)
            self.bias = bias
            self.scale = scale
        self.kwargs = kwargs
        self.bias.requires_grad = False
        self.scale.requires_grad = False
        return self.bias, self.scale

    def transform(self, data, offset=None):
        assert isinstance(data, (np.ndarray, torch.Tensor))
        if offset is None or self.scale.numel() == 1:
            bias = self.bias
            scale = self.scale
        else:
            assert offset.shape == (data.shape[0], 2)
            patch_size = data.shape[-2:]
            bias = torch.cat(
                [self.bias[:, :, x[0]:(x[0] + patch_size[0]), x[1]:(x[1] + patch_size[1])] for x in offset],
                dim=0
            )
            scale = torch.cat(
                [self.scale[:, :, x[0]:(x[0] + patch_size[0]), x[1]:(x[1] + patch_size[1])] for x in offset],
                dim=0
            )
        is_numpy_array = isinstance(data, np.ndarray)
        if is_numpy_array:
            if bias.numel() > 0:
                bias = bias.cpu().numpy()
                data = data - bias
            if scale.numel() > 0:
                scale = scale.cpu().numpy()
                data = data / scale
        else:
            if bias.numel() > 0:
                if data.device != bias.device:
                    bias = bias.to(data.device)
                data = data - bias
            if scale.numel() > 0:
                if data.device != scale.device:
                    scale = scale.to(data.device)
                data = data / scale
        return data

    def transform_back(self, data, offset=None):
        assert isinstance(data, (np.ndarray, torch.Tensor))
        if offset is None or self.scale.numel() == 1:
            bias = self.bias
            scale = self.scale
        else:
            assert offset.shape == (data.shape[0], 2)
            patch_size = data.shape[-2:]
            bias = torch.cat(
                [self.bias[:, :, x[0]:(x[0] + patch_size[0]), x[1]:(x[1] + patch_size[1])] for x in offset],
                dim=0
            )
            scale = torch.cat(
                [self.scale[:, :, x[0]:(x[0] + patch_size[0]), x[1]:(x[1] + patch_size[1])] for x in offset],
                dim=0
            )
        if isinstance(data, np.ndarray):
            if scale.numel() > 0:
                scale = scale.cpu().numpy()
                data = data * scale
            if bias.numel() > 0:
                bias = bias.cpu().numpy()
                data = data + bias
        else:
            if scale.numel() > 0:
                if data.device != scale.device:
                    scale = scale.to(data.device)
                data = data * scale
            if bias.numel() > 0:
                if data.device != bias.device:
                    bias = bias.to(data.device)
                data = data + bias
        return data

    def forward(self, data, mode=None):
        if mode == 'apply':
            data = self.transform(data)
        elif mode == 'revert':
            data = self.transform_back(data)
        else:
            raise NotImplementedError('[ERROR] Operation mode not available.')
        return data

    def compute_scalings_numpy(self, data: np.ndarray, **kwargs):
        raise NotImplementedError()

    def compute_scalings_torch(self, data: torch.Tensor, **kwargs):
        raise NotImplementedError()