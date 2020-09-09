import torch
import torch.nn as nn
from losses.loss_functions import LossFunction


class LossCollection(nn.Module):
    def __init__(self, *args):
        super(LossCollection, self).__init__()
        assert isinstance(args[0], LossFunction)
        self.objective_loss = args[0]
        self.complementary_losses = nn.ModuleList([])
        self.names = []
        self.use_mask = False
        self.use_scalings = False
        for i, arg in enumerate(args):
            assert isinstance(arg, LossFunction)
            self.complementary_losses.append(arg)
            self.names.append(arg.name)
            if arg.use_mask:
                self.use_mask = True
            if arg.use_scalings:
                self.use_scalings = True

    def objective(self, predictions, targets, mask=None, scalings=None, offset=None, ndim=2):
        if self.objective_loss.use_mask:
            assert mask is not None
        if self.objective_loss.use_scalings:
            return self.objective_loss(
                self._use_scalings(predictions, scalings, offset),
                self._use_scalings(targets, scalings, offset),
                mask=mask, ndim=ndim
            )
        else:
            return self.objective_loss(predictions, targets, mask=mask, ndim=ndim)

    def complementary(self, predictions, targets, mask=None, scalings=None, no_grad=True, offset=None, ndim=2):
        losses = []
        predictions_rescaled = None
        targets_rescaled = None
        if no_grad:
            with torch.no_grad():
                if self.use_scalings:
                    assert scalings is not None
                    predictions_rescaled = self._use_scalings(predictions, scalings, offset)
                    targets_rescaled = self._use_scalings(targets, scalings, offset)
                for loss in self.complementary_losses:
                    if loss.use_scalings:
                        losses.append(loss(predictions_rescaled, targets_rescaled, mask=mask, ndim=ndim))
                    else:
                        losses.append(loss(predictions, targets, mask=mask, ndim=ndim))
        else:
            if self.use_scalings:
                assert scalings is not None
                predictions_rescaled = self._use_scalings(predictions, scalings, offset)
                targets_rescaled = self._use_scalings(targets, scalings, offset)
            for loss in self.complementary_losses:
                if loss.use_scalings:
                    losses.append(loss(predictions_rescaled, targets_rescaled, mask=mask, ndim=ndim))
                else:
                    losses.append(loss(predictions, targets, mask=mask, ndim=ndim))
        return dict(zip(self.names, losses))

    def forward(self, predictions, targets, mask=None, scalings=None, no_grad=True, offset=None, ndim=2):
        obj = self.objective(predictions, targets, mask=mask, scalings=scalings, offset=offset, ndim=ndim)
        com = self.complementary(predictions, targets, mask=mask, scalings=scalings, no_grad=no_grad, offset=offset, ndim=ndim)
        return obj, com

    @staticmethod
    def _use_scalings(x, scalings, offset):
        output = x
        if scalings is not None and len(scalings) > 0:
            output = list(torch.split(output, [len(s.grids) for s in scalings], dim=1))
            for i, s in enumerate(scalings):
                if s.scaler is not None:
                    output[i] = s.scaler.transform_back(output[i], offset=offset)
            output = torch.cat(output, dim=1)
        return output

    def reset_batch_reduction(self, reduction_type):
        self.names = []
        for loss in self.complementary_losses:
            loss.reset_batch_reduction(reduction_type)
            self.names.append(loss.name)
        return self
