from training.localized_linear_model.TrainingEpoch import TrainingEpoch
from training.config_interface import BaseTrainingProcess


class TrainingProcess(BaseTrainingProcess):
    def __init__(self, config, grids, datasets, model, losses, optimizer, scheduler=None):
        super(TrainingProcess, self).__init__(
            config,
            grids,
            datasets,
            model, losses, optimizer, scheduler
        )

    def training_epoch(self):
        return TrainingEpoch(self)