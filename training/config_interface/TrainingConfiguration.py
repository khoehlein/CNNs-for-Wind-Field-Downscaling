import os
import warnings
import numpy as np
import random
import torch


class TrainingConfiguration(object):
    def __init__(
            self,
            directories,
            epochs,
            batch_size,
            initialization=None,
            residual_interpolation_mode=None,
            shuffling=None,
            saving_period=None,
            random_seed=None,
            device=None,
    ):
        self.directories = directories
        self.epochs = epochs
        self.batch_size = batch_size
        self.initialization = initialization
        self.residual_interpolation_mode = residual_interpolation_mode
        if shuffling is None:
            shuffling = False
        self.shuffling = shuffling
        self.saving_period = saving_period
        self.random_seed = random_seed
        self._set_device(device)

    def _set_device(self, device):
        if device is not None:
            if isinstance(device, (str, int)):
                os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
                self.device = torch.device('cuda:0')
            else:
                assert isinstance(device, torch.device)
                self.device = device
        else:
            if torch.cuda.is_available():
                print('[INFO] Using standard CUDA device for training.')
                self.device = torch.device('cuda:0')
            else:
                warnings.warn('[WARNING] CUDA device unavailable. Training is performed on CPU.')
                self.device = torch.device('cpu')

    def seed_random(self):
        # seed random number generators
        if self.random_seed is None:
            self.random_seed = random.randint(0, 2 ** 32 - 1)
        print("[INFO] Random seed: {}".format(self.random_seed))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)