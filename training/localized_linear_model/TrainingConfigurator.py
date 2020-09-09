from training.config_interface.TrainingConfiguration import TrainingConfiguration
from training.modular_downscaling_model.TrainingConfigurator import TrainingConfigurator as BaseConfigurator
from .TrainingProcess import TrainingProcess


class TrainingConfigurator(BaseConfigurator):
    def __init__(self):
        super(TrainingConfigurator, self).__init__()

    def build_training_process(self, grids, datasets, losses, model, config):
        training_config = TrainingConfiguration(
            self._prepare_output_directories(config, model),
            config["epochs"],
            config["batch_size"],
            config["initialization_mode"] if "initialization_mode" in config else None,
            None,
            config["shuffling"] if "shuffling" in config else True,
            config["saving_period"] if "saving_period" in config else None,
            config["random_seed"] if "random_seed" in config else None,
            config["gpu"] if "gpu" in config else None,
        )
        optimizer = self._prepare_optimizer(config['optimizer'], model)
        if 'scheduler' in config and config['scheduler'] is not None:
            scheduler = self._prepare_scheduler(config['scheduler'], optimizer)
        else:
            scheduler = None
        return TrainingProcess(training_config, grids, datasets, model, losses, optimizer, scheduler)