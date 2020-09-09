import numpy as np
from data.patching import PatchCutter
from .LowResHighResDataset import LowResHighResDataset


class PatchData(LowResHighResDataset):
    def __init__(self, dataset, patch_size_lr=None):
        assert isinstance(dataset, LowResHighResDataset)
        super(PatchData, self).__init__(
            dataset.geometry_lr,
            dataset.geometry_hr,
            dataset.grid_names_lr,
            dataset.grid_names_hr,
            dataset.grid_names_target
        )
        self.dataset = dataset
        self.patch_cutter_lr = PatchCutter(patch_size=patch_size_lr, dim=self.dim)
        patch_size_hr = np.array(self.scale_factor) * np.array(patch_size_lr)
        self.patch_cutter_hr = PatchCutter(patch_size=patch_size_hr, dim=self.dim)

    def __getitem__(self, item):
        target, input_lr, input_hr, mask_lr, mask_hr = self.dataset[item]
        self.randomize()
        target, _ = self.patch_cutter_hr(target)
        input_lr, _ = self.patch_cutter_lr(input_lr)
        input_hr, _ = self.patch_cutter_hr(input_hr)
        mask_lr, offset_lr = self.patch_cutter_lr(mask_lr)
        mask_hr, offset_hr = self.patch_cutter_hr(mask_hr)
        return target, input_lr, input_hr, mask_lr, mask_hr, offset_lr, offset_hr

    def __len__(self):
        return len(self.dataset)

    def randomize(self):
        self.patch_cutter_lr.randomize()
        self.patch_cutter_hr.synchronize(self.patch_cutter_lr)
        return self.patch_cutter_lr, self.patch_cutter_hr

    def get_input_lr(self, grid_names):
        return self.dataset.get_input_lr(grid_names)

    def get_input_hr(self, grid_names):
        return self.dataset.get_input_hr(grid_names)

    def get_target(self, grid_names):
        return self.dataset.get_target(grid_names)

    def set_input_lr(self, grid_names, data):
        return self.dataset.set_input_lr(grid_names, data)

    def set_input_hr(self, grid_names, data):
        return self.dataset.set_input_hr(grid_names, data)

    def set_target(self, grid_names, data):
        return self.dataset.set_target(grid_names, data)

    def grids(self):
        self.dataset.grids()
        return self

    def samples(self):
        self.dataset.samples()
        return self
