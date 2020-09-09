import numpy as np
import h5py
import pandas
from utils.ProgressBar import ProgressBar


class HDF5File:
    def __init__(self, filename, time_format='%Y-%m-%d %H'):
        print('[INFO]: Load information from H5 file <{}>'.format(filename))

        self._file = h5py.File(filename, "r")
        self._version = self._file.attrs['version']
        self.mode = self._file.attrs['mode']
        if not isinstance(self.mode, str):
            self.mode = self.mode.decode('utf-8')
        self.area = self._file.attrs['area']
        if not isinstance(self.area, str):
            self.area = self.area.decode('utf-8')

        self._time_format = time_format

        print('[INFO]: Convert timesteps <{}>'.format(filename))
        self._valid_times = pandas.to_datetime(
            self._file['valid_times'][...].astype(str),
            format=self._time_format,
        ).values
        self._dynamic_datasets = self._file['dynamic_variables']
        self._static_datasets = self._file['static_variables']

    def _compute_time_range(self):
        # take timesteps and compute min / max date range
        pass

    def is_timestep_in_data(self, timestep):
        #assert(isinstance(time, datetime.datetime))
        return timestep in self._valid_times

    def get_version(self):
        return self._version

    def get_static_datasets(self, datasets, raw=False):
        assert isinstance(datasets, (list, str))
        if not isinstance(datasets, list):
            datasets = [datasets]
        result = []
        grids_contained = [d for d in datasets if d in self._static_datasets.keys()]
        if len(grids_contained) > 0:
            progress = ProgressBar(len(grids_contained))
            print("[INFO]: Load static grids <{}> for <{}> ...".format(grids_contained, self.area))
            progress.proceed(0)
            for i, grid_name in enumerate(grids_contained):
                if grid_name == 'seaMask':
                    lsm = self._static_datasets[grid_name]
                    if not raw:
                        lsm = (np.array(lsm) > 0.5).astype(np.float32)
                    result += [lsm]
                else:
                    result += [self._static_datasets[grid_name]]
                progress.proceed(i + 1)
        return np.array(result), grids_contained

    def get_dynamic_datasets_at_index(self, datasets, index):
        if not isinstance(datasets, list):
            datasets = [datasets]
        result = []
        for grid in datasets:
            if grid in self._dynamic_datasets.keys():
                result += [self._dynamic_datasets[grid][index]]
            else:
                raise self._grid_not_found_warning(grid)
        return np.array(result)

    def _grid_not_found_warning(self, grid_name):
        return Warning('[WARNING] Grid <{}> not found.'.format(grid_name))

    def get_dynamic_datasets_at_timestep(self, datasets, timestep):
        idx = np.argwhere(self._valid_times == timestep).flatten()
        if len(idx) == 0:
            raise Exception('[EXCEPT] File did not contain valid timesteps.')
        return self.get_dynamic_datasets_at_index(datasets, idx)

    def get_dynamic_datasets_in_timerange(self, datasets, time_min, time_max):
        assert(isinstance(time_min, np.datetime64))
        assert(isinstance(time_max, np.datetime64))
        # find valid time stamps in range
        mask = np.logical_and(time_min <= self._valid_times, self._valid_times <= time_max)
        idxs = np.argwhere(mask).flatten().astype(np.int32)
        # remove duplicate time stamps
        valid_times = self._valid_times[idxs]
        valid_times, valid_idxs = np.unique(valid_times, return_index=True)
        idxs = idxs[valid_idxs]
        result = []
        # find all grids contained in dynamic variables
        grids_contained = [d for d in datasets if d in self._dynamic_datasets.keys()]
        if len(grids_contained) > 0:
            progress = ProgressBar(len(grids_contained))
            print("[INFO]: Load dynamic grids <{}> for <{}> in time range ({} - {}) ...".format(
                grids_contained, self.area, time_min, time_max)
            )
            progress.proceed(0)
            for i, grid in enumerate(grids_contained):
                # if grid in self._dynamic_datasets.keys():
                selected_grids = self._dynamic_datasets[grid]
                result += [selected_grids[idxs]]
                progress.proceed(i + 1)
        return np.array(result), self._valid_times[idxs], grids_contained

    def get_valid_times(self):
        return self._valid_times

    def get_dynamic_dataset_names(self):
        return self._dynamic_datasets.keys()

    def get_static_dataset_names(self):
        return self._static_datasets.keys()

    def close(self):
        self._file.close()
