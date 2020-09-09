import torch
from training.config_interface import BaseTrainingEpoch, BaseTrainingProcess
from utils import ProgressBar

import matplotlib.pyplot as plt


class TrainingEpoch(BaseTrainingEpoch):
    def __init__(self, training_process: BaseTrainingProcess):
        super(TrainingEpoch, self).__init__(training_process)

    def train(self):
        assert "training" in self.data_loaders
        # set cumulative losses to zero
        self.losses = self.training_process.losses['training'].to(self.training_process.config.device)
        self.cumulative_losses = {loss_name: 0 for loss_name in self.losses.names}
        if self.training_process.interpolator is not None:
            self.cumulative_losses.update({'itp_' + loss_name: 0 for loss_name in self.losses.names})
        # set mode of network to train (parameters will be affected / updated)
        self.training_process.model.train()
        # set progress bar
        progressBar = ProgressBar(self.num_loadings['training'], displaySumCount=True)
        # iterate over data loader
        for i, batch in enumerate(self.data_loaders['training']):
            # load data to device
            (
                targets_device,
                inputs_device,
                mask_lr_device, mask_hr_device,
                offset_lr, offset_hr
            ) = self._prepare_data(batch)
            # zero out optimizer weights
            self.training_process.optimizer.zero_grad()
            # apply model
            predictions_device, interpolate = self._apply_model(inputs_device, offset_lr, offset_hr)
            # compute loss to current targets
            loss = self._update_losses(predictions_device, targets_device, interpolate, mask_hr_device, offset_hr)
            # backpropagate loss
            loss.backward()
            # advance weights towards minimum
            self.training_process.optimizer.step()
            # update progress bar
            progressBar.proceed(i + 1)
        # update epoch count
        self._update_summary('training')
        # save epoch state
        if self.training_process.config.saving_period is not None:
            self._save_state()
        # update scheduler
        self._update_scheduler()

    def _print_grids(self, *args):
        for arg in args:
            plt.figure()
            plt.imshow(arg[0])
            plt.colorbar()
        plt.show()

    def validate(self):
        assert "validation" in self.data_loaders
        # set cumulative losses to 0
        if "validation" in self.training_process.losses:
            self.losses = self.training_process.losses['validation'].to(self.training_process.config.device)
        else:
            self.losses = self.training_process.losses['training'].to(self.training_process.config.device)
        self.cumulative_losses = {loss_name: 0 for loss_name in self.losses.names}
        if self.training_process.interpolator is not None:
            self.cumulative_losses.update({'itp_' + loss_name: 0 for loss_name in self.losses.names})
        # set mode of network to eval (parameters will not be affected / updated)
        self.training_process.model.eval()
        # set progress bar
        progressBar = ProgressBar(self.num_loadings['validation'], displaySumCount=True)
        # iterate over data loader
        with torch.no_grad():
            for i, batch in enumerate(self.data_loaders['validation']):
                # load data to device
                (
                    targets_device,
                    inputs_device,
                    mask_lr_device, mask_hr_device,
                    offset_lr, offset_hr
                ) = self._prepare_data(batch)
                # apply model
                predictions_device, interpolate = self._apply_model(inputs_device, offset_lr, offset_hr)
                # compute loss to current targets
                self._update_losses(predictions_device, targets_device, interpolate, mask_hr_device, offset_hr)
                # update progress bar
                progressBar.proceed(i + 1)
        # update epoch count
        self._update_summary('validation')

    def test(self):
        assert "test" in self.data_loaders
        # set cumulative losses to 0
        if "test" in self.training_process.losses:
            self.losses = self.training_process.losses['test'].to(self.training_process.config.device)
        else:
            self.losses = self.training_process.losses['training'].to(self.training_process.config.device)
        self.cumulative_losses = {loss_name: 0 for loss_name in self.losses.names}
        if self.training_process.interpolator is not None:
            self.cumulative_losses.update({'itp_' + loss_name: 0 for loss_name in self.losses.names})
        # set mode of network to eval (parameters will not be affected / updated)
        self.training_process.model.eval()
        # set progress bar
        progressBar = ProgressBar(self.num_loadings['test'], displaySumCount=True)
        # iterate over data loader
        with torch.no_grad():
            for i, batch in enumerate(self.data_loaders['test']):
                # load data to device
                (
                    targets_device,
                    inputs_device,
                    mask_lr_device, mask_hr_device,
                    offset_lr, offset_hr
                ) = self._prepare_data(batch)
                # apply model
                predictions_device, interpolate = self._apply_model(inputs_device, offset_lr, offset_hr)
                # compute loss to current targets
                self._update_losses(predictions_device, targets_device, interpolate, mask_hr_device, offset_hr)
                # update progress bar
                progressBar.proceed(i + 1)
        # update epoch count
        self._update_summary('test')

    def _prepare_data(self, batch):
        (
            target,
            input_lr, input_hr,
            mask_lr, mask_hr,
            *remaining
        ) = batch
        inputs = []
        if len(input_lr) > 0:
            inputs.append(input_lr.to(self.training_process.config.device))
        if len(input_hr) > 0:
            inputs.append(input_hr.to(self.training_process.config.device))
        if len(target) > 0:
            target = target.to(self.training_process.config.device)
        shape = mask_lr.shape[-2:]
        if len(mask_lr) > 0:
            mask_lr = mask_lr.to(self.training_process.config.device)
            mask_lr = (1. - mask_lr)
            mask_lr = mask_lr * ((shape[0] * shape[1]) / torch.sum(mask_lr, dim=[1, 2], keepdim=True))
        shape = mask_hr.shape[-2:]
        if len(mask_hr) > 0:
            mask_hr = mask_hr.to(self.training_process.config.device)
            mask_hr = (1. - mask_hr)
            mask_hr = mask_hr * ((shape[0] * shape[1]) / torch.sum(mask_hr, dim=[1, 2], keepdim=True))
        if len(remaining) > 0:
            assert len(remaining) == 2
            offset_lr, offset_hr = remaining
        else:
            offset_lr = None
            offset_hr = None
        return target, tuple(inputs), mask_lr, mask_hr, offset_lr, offset_hr

    def _apply_model(self, inputs_device, offset_lr, offset_hr):
        predictions_device = self.training_process.model(*inputs_device)
        interpolate = None
        if self.training_process.interpolator is not None:
            interpolate = self.training_process.interpolator(
                inputs_device[0],
                scalings_lr=self.training_process.grids.input_scalings_lr, offset_lr=offset_lr,
                scalings_hr=self.training_process.grids.target_scalings, offset_hr=offset_hr
            )
            predictions_device = predictions_device + interpolate
        return predictions_device, interpolate

    def _update_losses(self, predictions_device, targets_device, interpolate, mask_hr_device, offset_hr):
        loss, complementary = self.losses(
            predictions_device, targets_device,
            mask=mask_hr_device,
            scalings=self.training_process.grids.target_scalings,
            offset=offset_hr
        )
        for loss_name in complementary.keys():
            self.cumulative_losses.update({
                loss_name: self.cumulative_losses[loss_name] + complementary[loss_name].item()
            })
        if interpolate is not None:
            _, complementary = self.losses(
                interpolate, targets_device,
                mask=mask_hr_device,
                scalings=self.training_process.grids.target_scalings,
                offset=offset_hr
            )
            for loss_name in complementary.keys():
                self.cumulative_losses.update({
                    'itp_' + loss_name:
                        self.cumulative_losses['itp_' + loss_name] + complementary[loss_name].item()
                })
        return loss
