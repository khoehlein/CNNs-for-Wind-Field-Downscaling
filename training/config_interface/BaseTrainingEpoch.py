import os
import torch


class BaseTrainingEpoch(object):
    def __init__(self, training_process):
        self.training_process = training_process
        self.data_loaders = training_process.data_loaders()
        self.num_loadings = {
            step_name: len(self.data_loaders[step_name])
            for step_name in self.data_loaders.keys()
        }
        self.losses = None
        self.cumulative_losses = None

    def run(self):
        if "training" in self.data_loaders:
            self.train()
        if "validation" in self.data_loaders:
            self.validate()
        if "test" in self.data_loaders:
            self.test()

    def train(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def _prepare_data(self, *args, **kwargs):
        raise NotImplementedError()

    def _apply_model(self, *args, **kwargs):
        raise NotImplementedError()

    def _update_losses(self, *args, **kwargs):
        raise NotImplementedError()

    def _update_summary(self, mode='training'):
        # update epoch counter
        if mode == 'training':
            self.training_process.epochs_trained += 1
            current_lr = self._get_current_lr()
            self.training_process.summary.add_scalar(
                mode + '/params/lr',
                current_lr,
                self.training_process.epochs_trained
            )
            print(
                '[INFO] {} - {}, Current LR: {}'.format(
                    mode.capitalize(),
                    list(self.cumulative_losses.keys())[0],
                    current_lr
                )
            )
        elif mode == 'validation' or mode == 'test':
            print('[INFO] {}:'.format(mode.capitalize()))
        else:
            pass
        # update records per train step
        for loss_name in self.cumulative_losses.keys():
            self.cumulative_losses.update(
                {loss_name: self.cumulative_losses[loss_name] / len(self.training_process.datasets[mode])}
            )
            self.training_process.summary.add_scalar(
                '{}/loss/{}'.format(mode, loss_name),
                self.cumulative_losses[loss_name],
                self.training_process.epochs_trained,
            )
            print('\t- {:64s} {}'.format(loss_name + ':', self.cumulative_losses[loss_name]))
        self.training_process.summary.flush()

    def _get_current_lr(self):
        current_lr = self.training_process.optimizer.param_groups[0]['lr']
        return current_lr

    def _save_state(self):
        # do not save any network states if saveEveryNthEpoch is set to -1 (or any negative value)
        saving_period = self.training_process.config.saving_period
        if saving_period is not None and self.training_process.epochs_trained % saving_period == 0:
            model_file_name = os.path.join(
                self.training_process.config.directories['models'],
                'epoch_{}.pth'.format(self.training_process.epochs_trained)
            )
            state = {
                'epoch': self.training_process.epochs_trained,
                'model': self.training_process.model,
                'interpolator': self.training_process.interpolator,
            }
            state.update({
                'optimizer': self.training_process.optimizer,
                'scheduler': self.training_process.scheduler
            })
            state.update({
                'input_scalings_lr': self.training_process.grids.input_scalings_lr,
                'input_scalings_hr': self.training_process.grids.input_scalings_hr,
                'target_scalings': self.training_process.grids.target_scalings,
            })
            torch.save(state, model_file_name)
            print(
                "[INFO] Saved checkpoint for epoch {} to file {}.".format(
                    self.training_process.epochs_trained,
                    model_file_name
                )
            )

    def _update_scheduler(self):
        if isinstance(self.training_process.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.training_process.scheduler.step(
                self.cumulative_losses[
                    list(self.cumulative_losses.keys())[0]
                ]
            )
        if isinstance(self.training_process.scheduler, torch.optim.lr_scheduler.StepLR):
            self.training_process.scheduler.step(self.training_process.epochs_trained)


# class TemporalDownscalingEpoch(BaseTrainingEpoch):
#     def __init__(self, training_process, data_manager):
#         super(TemporalDownscalingEpoch, self).__init__(training_process, data_manager)
#         numStepsPast = self.training_process.config['coherence']['numStepsPast']
#         numStepsFuture = self.training_process.config['coherence']['numStepsFuture']
#         self.numSteps = numStepsPast + numStepsFuture + 1
#         self.selection = list(np.arange(self.numSteps))
#         self.selection.pop(numStepsPast)
#         self.idxPresent = numStepsPast
#         self.shapeInputs_expanded = (self.numSteps, batchSize, *shapeInputs)
#         self.shapeTargets_expanded = (self.numSteps, batchSize, *shapeTargets)
#         self.shapeInputs_compressed = (self.numSteps * batchSize, *shapeInputs)
#         self.shapeTargets_compressed = (self.numSteps * batchSize, *shapeTargets)
#
#     def train(self):
#         # set cumulative losses to zero
#         self._initialize_losses(self.training_process.training_losses)
#         # set mode of network to train (parameters will be affected / updated)
#         self.training_process.model.train()
#         # set progress bar
#         self.num_loadings = len(self.training_loader)
#         progressBar = ProgressBar(self.num_loadings, displaySumCount=True)
#         # iterate over data loader
#         for i, batch in enumerate(self.training_loader, 0):
#             # load data to device
#             inputs_device, targets_device, hrOro_device, mask_device = self._prepare_data(batch)
#             # zero out optimizer weights
#             self.training_process.optimizer.zero_grad()
#             # apply model
#             predictions_device = self._apply_model(inputs_device, hrOro_device)
#             # compute loss to current targets
#             loss = self._update_loss(predictions_device, targets_device, mask_device)
#             # backpropagate loss
#             loss.backward()
#             # advance weights towards minimum
#             self.training_process.optimizer.step()
#             # update progress bar
#             progressBar.proceed(i + 1)
#         # update epoch count
#         self._update_summary('training')
#         # save epoch state
#         self._save_state()
#         # update scheduler
#         self._update_scheduler()
#
#     def validate(self):
#         # set cumulative losses to 0
#         self._initialize_losses(self.training_process.validation_losses)
#         # set mode of network to train (parameters will be affected / updated)
#         self.training_process.model.eval()
#         # set progress bar
#         self.num_loadings = len(self.validation_loader)
#         progressBar = ProgressBar(self.num_loadings, displaySumCount=True)
#         # iterate over data loader
#         with torch.no_grad():
#             for i, batch in enumerate(self.validation_loader, 0):
#                 # load data to device
#                 inputs_device, targets_device, hrOro_device, mask_device = self._prepare_data(batch)
#                 # apply model
#                 predictions_device = self._apply_model(inputs_device, hrOro_device)
#                 # compute loss to current targets
#                 self._update_loss(predictions_device, targets_device, mask_device)
#                 # update progress bar
#                 progressBar.proceed(i + 1)
#         # update epoch count
#         self._update_summary('validation')
#
#     def _prepare_data(self, *args, **kwargs):
#         raise NotImplementedError()
#
#     def _apply_model(self, *args, **kwargs):
#         raise NotImplementedError()
#
#     def _update_loss(self, *args, **kwargs):
#         raise NotImplementedError()
#
#
# class ProbabilisticDownscalingEpoch(BaseTrainingEpoch):
#     def __init__(self, training_process, data_manager):
#         super(ProbabilisticDownscalingEpoch, self).__init__(
#             training_process, data_manager
#         )
#
#     def train(self):
#         # set cumulative losses to zero
#         self._initialize_losses(self.training_process.training_losses)
#         # set mode of network to train (parameters will be affected / updated)
#         self.training_process.model.train()
#         # set progress bar
#         self.num_loadings = len(self.training_loader)
#         progressBar = ProgressBar(self.num_loadings, displaySumCount=True)
#         # iterate over data loader
#         for i, batch in enumerate(self.training_loader, 0):
#             # load data to device
#             inputs_lr_device, targets_device, inputs_hr_device, mask_device = self._prepare_data(batch)
#             # zero out optimizer weights
#             self.training_process.optimizer.zero_grad()
#             # apply model
#             reconstruction_device, encoding_device = self._apply_model(
#                 targets_device, inputs_lr_device, inputs_hr_device
#             )
#             # query divergence weight
#             weight_divergence = self._get_divergence_weight('training')
#             # compute loss to current targets
#             loss = self._update_loss(
#                 targets_device, reconstruction_device, encoding_device,
#                 weight_divergence=weight_divergence,
#                 mask_device=mask_device
#             )
#             # backpropagate loss
#             loss.backward()
#             # advance weights towards minimum
#             self.training_process.optimizer.step()
#             # update progress bar
#             progressBar.proceed(i + 1)
#         # update epoch count
#         self._update_summary('training')
#         # save epoch state
#         self._save_state()
#         # update scheduler
#         self._update_scheduler()
#         # update divergence annealer
#         self._update_divergence_annealer()
#
#     def validate(self):
#         # set cumulative losses to 0
#         self._initialize_losses(self.training_process.validation_losses)
#         # set mode of network to train (parameters will be affected / updated)
#         self.training_process.model.eval()
#         # set progress bar
#         self.num_loadings = len(self.validation_loader)
#         progressBar = ProgressBar(self.num_loadings, displaySumCount=True)
#         # iterate over data loader
#         with torch.no_grad():
#             for i, batch in enumerate(self.validation_loader, 0):
#                 # load data to device
#                 inputs_lr_device, targets_device, inputs_hr_device, mask_device = self._prepare_data(batch)
#                 # apply model
#                 reconstruction_device, encoding_device = self._apply_model(
#                     targets_device, inputs_lr_device, inputs_hr_device
#                 )
#                 # query divergence weight
#                 weight_divergence = self._get_divergence_weight('validation')
#                 # compute loss to current targets
#                 loss = self._update_loss(
#                     targets_device, reconstruction_device, encoding_device,
#                     weight_divergence=weight_divergence,
#                     mask_device=mask_device
#                 )
#                 # update progress bar
#                 progressBar.proceed(i + 1)
#         # update epoch count
#         self._update_summary('validation')
#
#     def _prepare_data(self, *args, **kwargs):
#         raise NotImplementedError()
#
#     def _apply_model(self, *args, **kwargs):
#         raise NotImplementedError()
#
#     def _update_loss(self, *args, **kwargs):
#         raise NotImplementedError()
#
#     def _get_divergence_weight(self, mode='training'):
#         return self.training_process.divergence_annealer.get_eps(mode)
#
#     def _update_divergence_annealer(self):
#         self.training_process.divergence_annealer.step()
