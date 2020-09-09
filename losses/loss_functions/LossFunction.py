import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(
            self,
            name, use_mask=True, use_scalings=False,
            batch_reduction=None, spatial_reduction=None
    ):
        super(LossFunction, self).__init__()
        self.use_mask = use_mask
        self.use_scalings = use_scalings
        self.batch_reduction = batch_reduction
        if batch_reduction is None:
            self.batch_reduction = 'mean'
        self.spatial_reduction = spatial_reduction
        if spatial_reduction is None:
            self.spatial_reduction = 'mean'
        self._set_name(name)

    def local_deviation(self, predictions, targets, keepdim=True):
        raise NotImplementedError()

    def _reduce_batch(self, batch_deviation, keepdim=True):
        if self.batch_reduction == 'mean':
            batch_deviation = torch.mean(batch_deviation, dim=0, keepdim=keepdim)
        elif self.batch_reduction == 'sum':
            batch_deviation = torch.sum(batch_deviation, dim=0, keepdim=keepdim)
        elif self.batch_reduction == 'none':
            if not keepdim:
                raise Exception('[ERROR] Omitting batch reduction while not keeping dimensions results in errors.')
        else:
            raise NotImplementedError()
        return batch_deviation

    def _reduce_space(self, batch_local_deviation, ndim=2, keepdim=True):
        dim = len(batch_local_deviation.size())
        assert dim >= ndim
        dims = tuple(range(dim - ndim, dim))
        if self.spatial_reduction == 'mean':
            batch_deviation = torch.mean(batch_local_deviation, dim=dims, keepdim=keepdim)
        elif self.spatial_reduction == 'sum':
            batch_deviation = torch.sum(batch_local_deviation, dim=dims, keepdim=keepdim)
        elif self.spatial_reduction == 'none':
            batch_deviation = batch_local_deviation
        else:
            raise NotImplementedError()
        return batch_deviation

    def _apply_mask(self, batch_local_deviation, mask):
        assert mask is not None
        assert len(mask.size()) == 3
        assert len(batch_local_deviation.size()) == 4
        return batch_local_deviation * mask.unsqueeze(dim=1)

    def forward(self, predictions, targets, mask=None, keepdim=True, ndim=2, flatten_result=True):
        dev = self.local_deviation(predictions, targets, keepdim=keepdim)
        if self.use_mask:
            dev = self._apply_mask(dev, mask)
        dev = self._reduce_space(dev, keepdim=keepdim, ndim=ndim)
        dev = self._reduce_batch(dev, keepdim=keepdim)
        if not flatten_result:
            return dev
        else:
            return dev.flatten()

    def _set_name(self, name):
        self.name = name
        if self.use_mask:
            self.name += '_masked'
        if self.use_scalings:
            self.name += '_scaled'
        self.name += "_spatial-{}".format(self.spatial_reduction)
        self.name += "_batch-{}".format(self.batch_reduction)

    def reset_batch_reduction(self, reduction_type):
        assert reduction_type in ['mean', 'sum', 'none']
        self.batch_reduction = reduction_type
        self._set_name(self.name.split('_')[0])
        return self
