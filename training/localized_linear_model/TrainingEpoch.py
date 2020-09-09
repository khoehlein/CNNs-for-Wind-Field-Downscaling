import torch
from training.modular_downscaling_model.TrainingEpoch import TrainingEpoch as BaseEpoch


class TrainingEpoch(BaseEpoch):
    def __init__(self, training_process):
        super(TrainingEpoch, self).__init__(training_process=training_process)

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
        inputs = torch.cat(inputs, dim=-1)
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
        return target, inputs, mask_lr, mask_hr, offset_lr, offset_hr

    def _apply_model(self, inputs_device, offset_lr, offset_hr):
        predictions_device = self.training_process.model(inputs_device)
        interpolate = None
        return predictions_device, interpolate

    def _update_losses(self, predictions_device, targets_device, interpolate, mask_hr_device, offset_hr):
        loss, complementary = self.losses(
            predictions_device, targets_device,
            mask=mask_hr_device,
            scalings=self.training_process.grids.target_scalings,
            offset=offset_hr,
            ndim=1
        )
        for loss_name in complementary.keys():
            self.cumulative_losses.update({
                loss_name: self.cumulative_losses[loss_name] + complementary[loss_name].item()
            })
        return loss