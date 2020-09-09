import numpy as np
import os
import glob

from data.importing.Importer import Importer
from data.importing.hdf5.HDF5File import HDF5File


class HDF5Importer(Importer):
    def __init__(self, directory, low_res_filter_name='input', high_res_filter_name='target'):
        self.low_res_filter_name = low_res_filter_name
        self.high_res_filter_name = high_res_filter_name
        super(HDF5Importer, self).__init__(directory)

    def _load_files(self):
        all_files = sorted(glob.glob(self.directory + "/**/*.hdf5", recursive=True))

        if len(all_files) == 0:
            raise Exception("Dataset directory <{}> is empty.".format(self.directory))

        # divide into low_res and high_res files (ERA5 = low-res, HRES = high-res)
        low_res_files = list(filter(lambda x: os.path.basename(x).startswith(self.low_res_filter_name), all_files))
        high_res_files = list(filter(lambda x: os.path.basename(x).startswith(self.high_res_filter_name), all_files))

        assert (len(low_res_files) == len(high_res_files))

        # load all low_res HDF5 files
        for filename in low_res_files:
            file = HDF5File(filename)
            # print('[INFO]: Open low_res HDF5 file <{}>'.format(filename))
            cur_area = file.area
            self.preloaded_files_low_res += [(file, cur_area)]

        # load all high_res HDF5 files
        for filename in high_res_files:
            file = HDF5File(filename)
            # print('[INFO]: Open high_res HDF5 file <{}>'.format(filename))
            cur_area = file.area
            self.preloaded_files_high_res += [(file, cur_area)]

    def get_dynamic_grids_training_pair(self, grids_low_res, grids_high_res, time_min, time_max, area):
        low_res_grids, low_res_timesteps, low_res_grid_names = self.get_low_res_dynamic_grids(
            grids_low_res, time_min, time_max, area
        )
        high_res_grids, high_res_timesteps, high_res_grid_names = self.get_high_res_dynamic_grids(
            grids_high_res, time_min, time_max, area
        )

        assert(len(low_res_timesteps) == len(high_res_timesteps))

        return (
            [low_res_grids, low_res_timesteps, low_res_grid_names],
            [high_res_grids, high_res_timesteps, high_res_grid_names]
        )

    def get_static_grids_training_pair(self, grids_low_res, grids_high_res, area):
        low_res_grids = self.get_low_res_static_grids(grids_low_res, area)
        high_res_grids = self.get_high_res_static_grids(grids_high_res, area)

        # assert (len(low_res_grids) == len(high_res_grids))

        return low_res_grids, high_res_grids

    def get_low_res_dynamic_grids(self, grids, time_min, time_max, area):
        return self._dynamic_grids(self.preloaded_files_low_res, time_min, time_max, area, grids)

    def get_low_res_static_grids(self, grids, area, raw=False):
        return self._static_grids(self.preloaded_files_low_res, area, grids, raw=raw)

    def get_high_res_dynamic_grids(self, grids, time_min, time_max, area):
        return self._dynamic_grids(self.preloaded_files_high_res, time_min, time_max, area, grids)

    def get_high_res_static_grids(self, grids, area, raw=True):
        return self._static_grids(self.preloaded_files_high_res, area, grids, raw=raw)

    def _dynamic_grids(self, files, time_min, time_max, area, grids):
        dynamic_grids = []
        all_timesteps = []
        grid_names = []
        for file, file_area in files:
            if file_area == area:
                grids, timesteps, grid_names = file.get_dynamic_datasets_in_timerange(grids, time_min, time_max)
                if len(grid_names) > 0:
                    dynamic_grids += [grids]
                all_timesteps += [timesteps]
        if len(dynamic_grids) > 0:
            dynamic_grids = np.concatenate(dynamic_grids, axis=1)
        if len(all_timesteps) > 0:
            all_timesteps = np.concatenate(all_timesteps, axis=0)
        return dynamic_grids, all_timesteps, grid_names

    def _static_grids(self, files, area, grids, raw=False):
        static_grids = []
        grid_names = []
        for file, file_area in files:
            if file_area == area:
                grids, grid_names = file.get_static_datasets(grids, raw=raw)
                if len(grid_names) > 0:
                    static_grids += [grids]
        if len(static_grids) > 0:
            static_grids = np.concatenate(static_grids, axis=0)
        return static_grids, grid_names

    # def get_training_pairs(self, time_min, time_max, area, grids_low_res, grids_high_res):
    #     pass
