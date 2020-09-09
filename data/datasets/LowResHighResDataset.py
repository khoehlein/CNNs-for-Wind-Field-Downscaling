import numpy as np
from torch.utils.data import Dataset
from collections import namedtuple


region_geometry = namedtuple('region_geometry', ['lon', 'lat', 'mask'])


class LowResHighResDataset(Dataset):
    def __init__(
            self,
            geometry_lr,
            geometry_hr,
            grid_names_lr,
            grid_names_hr,
            grid_names_target,
    ):
        assert isinstance(geometry_lr, (region_geometry, dict))
        assert isinstance(geometry_hr, (region_geometry, dict))
        self.geometry_lr = geometry_lr
        self.geometry_hr = geometry_hr
        if isinstance(geometry_lr, dict):
            geometry_lr = geometry_lr[list(geometry_lr.keys())[0]]
            geometry_hr = geometry_hr[list(geometry_hr.keys())[0]]
        self.shape_lr = geometry_lr[0].shape
        self.shape_hr = geometry_hr[0].shape
        assert len(self.shape_lr) == len(self.shape_hr)
        self.dim = len(self.shape_lr)
        scale_factor = np.array(self.shape_hr) / np.array(self.shape_lr)
        self.scale_factor = tuple(scale_factor)
        assert isinstance(grid_names_lr, dict)
        assert 'dynamic' in grid_names_lr and 'static' in grid_names_lr
        assert isinstance(grid_names_hr, dict)
        assert 'dynamic' in grid_names_hr and 'static' in grid_names_hr
        assert isinstance(grid_names_target, dict)
        assert 'dynamic' in grid_names_target and 'static' in grid_names_target
        self.grid_names_lr = grid_names_lr
        self.grid_names_hr = grid_names_hr
        self.grid_names_target = grid_names_target

    def input_grids_lr(self):
        return self.grid_names_lr['dynamic'] + self.grid_names_lr['static']

    def input_grids_hr(self):
        return self.grid_names_hr['dynamic'] + self.grid_names_hr['static']

    def target_grids(self):
        return self.grid_names_target['dynamic'] + self.grid_names_target['static']

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def grids(self):
        raise NotImplementedError()

    def samples(self):
        raise NotImplementedError()

    def get_input_lr(self, grid_names):
        raise NotImplementedError()

    def get_input_hr(self, grid_names):
        raise NotImplementedError()

    def get_target(self, grid_names):
        raise NotImplementedError()

    def set_input_lr(self, grid_names, data):
        raise NotImplementedError()

    def set_input_hr(self, grid_names, data):
        raise NotImplementedError()

    def set_target(self, grid_names, data):
        raise NotImplementedError()
