from training.modular_downscaling_model import TrainingEpoch
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
