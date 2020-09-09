import os
import numpy as np
import json

from torch.nn import init
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from .TrainingConfiguration import TrainingConfiguration
from data import GridConfiguration
from networks.modular_downscaling_model.input_modules import InterpolationModuleLR

from utils import NumpyEncoder


class BaseTrainingProcess(object):
    def __init__(
            self,
            config: TrainingConfiguration,
            grids: GridConfiguration,
            datasets,
            model,
            losses,
            optimizer,
            scheduler=None,
    ):
        self.config = config
        self.grids = grids
        self.datasets = datasets
        self.model = model
        if config.residual_interpolation_mode is not None:
            assert np.all([channel is not None for channel in self.grids.target_channels_in_lr]), \
                '[ERROR] Residual interpolation requires all target grids to be included in low-res inputs'
            self.interpolator = InterpolationModuleLR.from_config({
                'scale_factor': list(self.datasets.values())[0].scale_factor,
                'interpolation_mode': self.config.residual_interpolation_mode,
                'target_channels_in_lr': self.grids.target_channels_in_lr
            }, input_channels = len(self.grids.input_grids_lr))
        else:
            self.interpolator = None
        self.losses = losses
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summary = SummaryWriter(self.config.directories['records'], purge_step=1)
        self.epochs_trained = 0

    def run(self):
        self.config.seed_random()
        self.grids.to(self.config.device)
        self.model.to(self.config.device)
        if self.config.initialization is not None:
            self._initialize_model()
        for epoch in range(self.config.epochs):
            print("Epoch {} / {}".format(epoch + 1, self.config.epochs))
            epoch = self.training_epoch()
            epoch.run()
            if hasattr(epoch, 'terminous_condition'):
                if epoch.terminous_condition():
                    epoch.save_state()
                    return

    def training_epoch(self):
        raise NotImplementedError()

    def data_loaders(self):
        data_loaders = {
            step_name: DataLoader(
                self.datasets[step_name],
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffling if step_name == 'training' else False,
                drop_last=False,
                num_workers=0
            )
            for step_name in self.datasets.keys()
        }
        return data_loaders

    def _initialize_model(self, gain=0.02):
        initialization_method = self.config.initialization

        def initialization(m):
            class_name = m.__class__.__name__
            if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
                if initialization_method == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif initialization_method == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif initialization_method == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif initialization_method == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('Weight initialization method [{}] is not implemented.'.format(initialization_method))

                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif class_name.find('BatchNorm') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print("[INFO] Initializing network with method <{}>.".format(initialization_method))
        self.model.apply(initialization)

    def save_config(self, config):
        output_file_path = os.path.join(self.config.directories['records'], 'config.json')
        with open(output_file_path, 'w') as f:
            json.dump(config, f, indent=4, cls=NumpyEncoder)
        return output_file_path
