import os
import sys

import torch.optim as optim

from training.config_interface.TrainingConfiguration import TrainingConfiguration
from .TrainingProcess import TrainingProcess


example_config = {
    "training": {
        "directory": "results",
        "model_initialization": "orthogonal",
        "learn_residuals": False,
        "gpu": 0,
        "epochs": 10,
        "batch_size": 20,
        "saving_period": 2,
        "optimizer": {
            "type": "adam",
            "learningRate": 0.001,
            "betas": [0.9, 0.999],
            "weightDecay": 0.0001
        },
        "scheduler": {
            "type": "plateau",
            "gamma": 0.1,
            "steps": 5
        },
        "debugging": {
            "print_network_details": True,
            "print_config": False,
            "print_directories": False
        }
    },
}


class TrainingConfigurator(object):
    def __init__(self):
        pass

    def build_training_process(self, grids, datasets, losses, model, config):
        training_config = TrainingConfiguration(
            self._prepare_output_directories(config, model),
            config["epochs"],
            config["batch_size"],
            config["initialization_mode"] if "initialization_mode" in config else None,
            config["residual_interpolation_mode"] if "residual_interpolation_mode" in config else None,
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

    def _prepare_output_directories(self, config, model):
        output_directory = config["directory"]
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        if "id" in config and config["id"] is not None:
            training_name = config["id"]
        elif hasattr(model, 'name'):
            training_name = model.name
        else:
            training_name = "_missing_id"
        output_directory = os.path.join(output_directory, training_name)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        run_name, run_number = self._get_run_name(output_directory)
        output_directory = os.path.join(output_directory, run_name)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        directories = {
            'records': os.path.join(output_directory, 'records'),
            'models': os.path.join(output_directory, 'models'),
        }
        for dir_name in directories.keys():
            if not os.path.isdir(directories[dir_name]):
                os.makedirs(directories[dir_name])
        if 'debugging' in config:
            if config['debugging']['print_config']:
                print("[INFO] Config options:\n===================")
                self._print_configs(config)
                print("===================")
            if config['debugging']['print_directories']:
                print("[INFO] Directories::\n===================")
                self._print_configs(directories)
                print("===================")
        if run_number is not None:
            print("[INFO] Training run number: <{}>".format(int(run_number)))
        else:
            print("[INFO] Training in DEBUG mode.")
        return directories

    @staticmethod
    def _get_run_name(directory):
        if sys.gettrace():
            print("[WARNING]: Programs runs in DEBUG mode")
            run_name = 'run_debug'
            i = None
        else:
            taken_names = sorted(os.listdir(directory))
            i = 0
            while "run_{:05d}".format(i) in taken_names:
                i += 1
            run_name = "run_{:05d}".format(i)
        return run_name, i

    @staticmethod
    def _prepare_optimizer(config, model):
        optimizer_type = config['type']
        if optimizer_type == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                betas=config['betas'],
                weight_decay=(
                    config['weight_decay'] if 'weight_decay' in config else 0.0
                )
            )
        else:
            raise NotImplementedError()
        return optimizer

    @staticmethod
    def _prepare_scheduler(config, optimizer):
        scheduler_type = config['type']
        if scheduler_type is None:
            scheduler = None
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['steps'],
                gamma=config['gamma']
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config['gamma'],
                patience=config['steps'],
                verbose=False,
                threshold=1.e-2,
                threshold_mode='rel',
                cooldown=0, min_lr=1.e-6, eps=1e-08
            )
        else:
            raise NotImplementedError('Scheduler type <{}> is not implemented'.format(scheduler_type))
        return scheduler

    def _print_configs(self, config, indent=0):
        for key, value in config.items():
            # print('\t' * indent + str(key))
            if isinstance(value, dict):
                print('\t' * indent + str(key))
                print('\t' * indent + "{")
                self._print_configs(value, indent + 1)
                print('\t' * indent + "}")
            else:
                print('\t' * indent + '{} -> {}'.format(key, value))
