import numpy as np
from itertools import chain
from .DataSection import DataSection


class RandomShufflingDataSection(DataSection):
    def __init__(
            self,
            importer,
            region, time_range,
            input_grids_lr=None, input_grids_hr=None,
            target_grids=None,
            shuffling_grids_lr=None, shuffling_grids_hr=None,
            num_patches=None
    ):
        super(RandomShufflingDataSection, self).__init__(
            importer,
            region, time_range,
            input_grids_lr, input_grids_hr,
            target_grids
        )
        self.shuffling_names_lr = {}
        self._shuffling_index_lr = {}
        self.reset_shuffling_grids_lr(shuffling_grids_lr)
        self.shuffling_names_hr = {}
        self._shuffling_index_hr = {}
        self.reset_shuffling_grids_hr(shuffling_grids_hr)
        if num_patches is None:
            num_patches = [1, 1]
        else:
            if isinstance(num_patches, (int, float)):
                num_patches = (int(num_patches),) * 2
        self.num_patches = num_patches
        assert len(num_patches) == len(self.shape_lr)
        assert len(num_patches) == len(self.shape_hr)
        assert np.all(np.array([a % b for a, b in zip(self.shape_lr, num_patches)]) == 0)
        assert np.all(np.array([a % b for a, b in zip(self.shape_hr, num_patches)]) == 0)
        self.shuffling_index = None
        self.shuffle_samples()

    def reset_shuffling_grids_lr(self, shuffling_grids_lr=None):
        if shuffling_grids_lr is None:
            shuffling_grids_lr = []
        else:
            if not isinstance(shuffling_grids_lr, (list, tuple)):
                shuffling_grids_lr = [shuffling_grids_lr]
        self.shuffling_names_lr = {
            key: [grid_name for grid_name in shuffling_grids_lr if grid_name in self.grid_names_lr[key]]
            for key in self.grid_names_lr.keys()
        }
        assert len(self.shuffling_grids_lr()) == len(shuffling_grids_lr)
        self._shuffling_index_lr = {
            key: [self.grid_names_lr[key].index(grid_name) for grid_name in self.shuffling_names_lr[key]]
            for key in self.grid_names_lr.keys()
        }
        return self

    def reset_shuffling_grids_hr(self, shuffling_grids_hr=None):
        if shuffling_grids_hr is None:
            shuffling_grids_hr = []
        else:
            if not isinstance(shuffling_grids_hr, (list, tuple)):
                shuffling_grids_hr = [shuffling_grids_hr]
        self.shuffling_names_hr = {
            key: [grid_name for grid_name in shuffling_grids_hr if grid_name in self.grid_names_hr[key]]
            for key in self.grid_names_hr.keys()
        }
        assert len(self.shuffling_grids_hr()) == len(shuffling_grids_hr)
        self._shuffling_index_hr = {
            key: [self.grid_names_hr[key].index(grid_name) for grid_name in self.shuffling_names_hr[key]]
            for key in self.grid_names_hr.keys()
        }
        return self

    def shuffling_grids_lr(self):
        return self.shuffling_names_lr['dynamic'] + self.shuffling_names_lr['static']

    def shuffling_grids_hr(self):
        return self.shuffling_names_hr['dynamic'] + self.shuffling_names_hr['static']

    def shuffle_samples(self):
        self.shuffling_index = np.arange(0, self.__len__()).astype(np.int32)
        np.random.shuffle(self.shuffling_index)
        return self

    def __getitem__(self, item):
        if self._in_grid_mode:
            raise Exception('[ERROR] Item selection only supported in sample mode.')
        # construct target
        target = [self.target['dynamic'][item].copy()]
        if self.target['static'] is not None:
            target.append(self.target['static'][0])
        target = np.concatenate(target, axis=0)
        # Construct lr input
        input_lr = []
        if self.input_lr['dynamic'] is not None:
            input_dyn = self.input_lr['dynamic'][item].copy()
            if len(self._shuffling_index_lr['dynamic']) > 0:
                array_data = self.input_lr['dynamic'][self.shuffling_index[item], self._shuffling_index_lr['dynamic']]
                array_data = self._shuffle_patches(array_data)
                input_dyn[self._shuffling_index_lr['dynamic']] = array_data
            input_lr.append(input_dyn)
        if self.input_lr['static'] is not None:
            input_stat = self.input_lr['static'][0].copy()
            if len(self._shuffling_index_lr['static']) > 0:
                array_data = input_stat[self._shuffling_index_lr['static']]
                array_data = self._shuffle_patches(array_data)
                input_stat[self._shuffling_index_lr['static']] = array_data
            input_lr.append(input_stat)
        if len(input_lr) > 0:
            input_lr = np.concatenate(input_lr, axis=0)
        # Construct hr input
        input_hr = []
        if self.input_hr['dynamic'] is not None:
            input_dyn = self.input_hr['dynamic'][item].copy()
            if len(self._shuffling_index_hr['dynamic']) > 0:
                array_data = self.input_hr['dynamic'][self.shuffling_index[item], self._shuffling_index_hr['dynamic']]
                array_data = self._shuffle_patches(array_data)
                input_dyn[self._shuffling_index_hr['dynamic']] = array_data
            input_hr.append(input_dyn)
        if self.input_hr['static'] is not None:
            input_stat = self.input_hr['static'][0].copy()
            if len(self._shuffling_index_hr['static']) > 0:
                array_data = input_stat[self._shuffling_index_hr['static']]
                array_data = self._shuffle_patches(array_data)
                input_stat[self._shuffling_index_hr['static']] = array_data
            input_hr.append(input_stat)
        if len(input_hr) > 0:
            input_hr = np.concatenate(input_hr, axis=0)
        mask_lr = self.geometry_lr.mask
        mask_hr = self.geometry_hr.mask
        return target, input_lr, input_hr, mask_lr, mask_hr

    def _shuffle_patches(self, array_data):
        rows = list(np.split(array_data, self.num_patches[0], axis=1))
        patches = list(
            chain.from_iterable([
                np.split(row, self.num_patches[1], axis=2) for row in rows
            ])
        )
        np.random.shuffle(patches)
        rows = [
            np.concatenate(patches[i:(i+self.num_patches[1])], axis=2)
            for i in range(0, len(patches), self.num_patches[1])
        ]
        array_data = np.concatenate(rows, axis=1)
        return array_data
