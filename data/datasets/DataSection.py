import numpy as np
import datetime
from calendar import monthrange

from .LowResHighResDataset import region_geometry, LowResHighResDataset


class DataSection(LowResHighResDataset):
    def __init__(
            self,
            importer, region, time_range,
            input_grids_lr=None, input_grids_hr=None,
            target_grids=None,
    ):
        self.region = region
        self.time_range = time_range
        self.input_lr = {}
        self.input_hr = {}
        self.target = {}
        self.grid_names_lr = {}
        self.grid_names_hr = {}
        self.grid_names_target = {}
        self._load_data(importer, input_grids_lr, input_grids_hr, target_grids)
        super(DataSection, self).__init__(
            self.geometry_lr,
            self.geometry_hr,
            self.grid_names_lr,
            self.grid_names_hr,
            self.grid_names_target,
        )
        self._in_grid_mode = False

    def __len__(self):
        return 0 if self.time_stamps is None else len(self.time_stamps)

    def __getitem__(self, item):
        if self._in_grid_mode:
            raise Exception('[ERROR] Item selection only supported in sample mode.')
        target = [self.target['dynamic'][item]]
        if self.target['static'] is not None:
            target.append(self.target['static'][0])
        target = np.concatenate(target, axis=0)
        input_lr = [self.input_lr['dynamic'][item]]
        if self.input_lr['static'] is not None:
            input_lr.append(self.input_lr['static'][0])
        if len(input_lr) > 0:
            input_lr = np.concatenate(input_lr, axis=0)
        else:
            input_lr = []
        input_hr = []
        if self.input_hr['dynamic'] is not None:
            input_hr.append(self.input_hr['dynamic'][item])
        if self.input_hr['static'] is not None:
            input_hr.append(self.input_hr['static'][0])
        if len(input_hr) > 0:
            input_hr = np.concatenate(input_hr, axis=0)
        else:
            input_hr = []
        mask_lr = self.geometry_lr.mask
        mask_hr = self.geometry_hr.mask
        return target, input_lr, input_hr, mask_lr, mask_hr

    def get_input_lr(self, grid_names):
        return self._get_grid_data(self.input_lr, self.grid_names_lr, grid_names)

    def get_input_hr(self, grid_names):
        return self._get_grid_data(self.input_hr, self.grid_names_hr, grid_names)

    def get_target(self, grid_names):
        return self._get_grid_data(self.target, self.grid_names_target, grid_names)

    def set_input_lr(self, grid_names, data):
        return self._set_grid_data(self.input_lr, self.grid_names_lr, data, grid_names)

    def set_input_hr(self, grid_names, data):
        return self._set_grid_data(self.input_hr, self.grid_names_hr, data, grid_names)

    def set_target(self, grid_names, data):
        return self._set_grid_data(self.target, self.grid_names_target, data, grid_names)

    def grids(self):
        if not self._in_grid_mode:
            self._split_data_dicts(self.input_lr, self.grid_names_lr)
            self._split_data_dicts(self.input_hr, self.grid_names_hr)
            self._split_data_dicts(self.target, self.grid_names_target)
            self._in_grid_mode = True
        return self

    def samples(self):
        if self._in_grid_mode:
            self._combine_data_dicts(self.input_lr)
            self._combine_data_dicts(self.input_hr)
            self._combine_data_dicts(self.target)
            self._in_grid_mode = False
        return self

    def _load_data(self, importer, input_grids_lr, input_grids_hr, target_grids):
        self._load_dynamic_data(importer, input_grids_lr, input_grids_hr, target_grids)
        self._load_static_data(importer, input_grids_lr, input_grids_hr, target_grids)
        self._load_geometry(importer)

    def _load_dynamic_data(self, importer, input_grids_lr, input_grids_hr, target_grids):
        min_date, max_date = self._to_datetime_range(self.time_range)
        grids_lr, grids_hr = importer.get_dynamic_grids_training_pair(
            input_grids_lr, input_grids_hr,
            np.datetime64(min_date), np.datetime64(max_date),
            self.region
        )
        dynamic_input_lr, self.time_stamps, dynamic_input_names_lr = grids_lr
        if len(dynamic_input_names_lr) > 0:
            dynamic_input_lr = np.swapaxes(dynamic_input_lr, 0, 1)
        else:
            dynamic_input_lr = None
        self.input_lr.update({'dynamic': dynamic_input_lr})
        self.grid_names_lr.update({'dynamic': dynamic_input_names_lr})
        dynamic_input_hr, _, dynamic_input_names_hr = grids_hr
        if len(dynamic_input_names_hr) > 0:
            dynamic_input_hr = np.swapaxes(dynamic_input_hr, 0, 1)
        else:
            dynamic_input_hr = None
        self.input_hr.update({'dynamic': dynamic_input_hr})
        self.grid_names_hr.update({'dynamic': dynamic_input_names_hr})
        _, grids_hr = importer.get_dynamic_grids_training_pair(
            [], target_grids,
            np.datetime64(min_date), np.datetime64(max_date),
            self.region
        )
        dynamic_target, _, dynamic_target_names = grids_hr
        if len(dynamic_target_names) > 0:
            dynamic_target = np.swapaxes(dynamic_target, 0, 1)
        else:
            dynamic_target = 0
        self.target.update({'dynamic': dynamic_target})
        self.grid_names_target.update({'dynamic': dynamic_target_names})
        assert dynamic_input_lr is not None
        assert dynamic_target is not None

    def _load_static_data(self, importer, input_grids_lr, input_grids_hr, target_grids):
        grids_lr, grids_hr = importer.get_static_grids_training_pair(
            input_grids_lr, input_grids_hr,
            self.region
        )
        static_input_lr, static_input_names_lr = grids_lr
        if len(static_input_names_lr) > 0:
            static_input_lr = np.expand_dims(static_input_lr, axis=0)
        else:
            static_input_lr = None
        self.input_lr.update({'static': static_input_lr})
        self.grid_names_lr.update(({'static': static_input_names_lr}))
        static_input_hr, static_input_names_hr = grids_hr
        if len(static_input_names_hr) > 0:
            static_input_hr = np.expand_dims(static_input_hr, axis=0)
        else:
            static_input_hr = None
        self.input_hr.update({'static': static_input_hr})
        self.grid_names_hr.update(({'static': static_input_names_hr}))
        _, grids_hr = importer.get_static_grids_training_pair(
            [], target_grids,
            self.region
        )
        static_target, static_target_names = grids_hr
        if len(static_target_names) > 0:
            static_target = np.expand_dims(static_target, axis=0)
        else:
            static_target = None
        self.target.update({'static': static_target})
        self.grid_names_target.update(({'static': static_target_names}))

    def _load_geometry(self, importer):
        geometry_grids = ['lons', 'lats', 'padding_mask']
        grids_lr, grids_hr = importer.get_static_grids_training_pair(
            geometry_grids, geometry_grids,
            self.region
        )
        self.geometry_lr = region_geometry(grids_lr[0][0], grids_lr[0][1], grids_lr[0][2])
        self.geometry_hr = region_geometry(grids_hr[0][0], grids_hr[0][1], grids_hr[0][2])

    def _get_grid_data(self, data_dict, names_dict, grid_names):
        if not isinstance(grid_names, (list, tuple)):
            assert isinstance(grid_names, str)
            grid_names = [grid_names]
        dynamic_grids_requested = [(True if grid_name in names_dict['dynamic'] else False) for grid_name in grid_names]
        dynamic_grids_requested = np.any(dynamic_grids_requested)
        grid_data = []
        for grid_name in grid_names:
            if grid_name in names_dict['static']:
                channel = names_dict['static'].index(grid_name)
                current_grid_data = data_dict['static'][:, [channel]]
                if dynamic_grids_requested:
                    current_grid_data = np.repeat(current_grid_data, len(self), axis=0)
            elif grid_name in names_dict['dynamic']:
                channel = names_dict['dynamic'].index(grid_name)
                current_grid_data = data_dict['dynamic'][:, [channel]]
            else:
                raise Exception('[ERROR] Grid <{}> not found.'.format(grid_name))
            grid_data.append(current_grid_data)
        grid_data = np.concatenate(grid_data, axis=1)
        return grid_data, grid_names

    def _set_grid_data(self, data_dict, names_dict, grid_data, grid_names):
        if not isinstance(grid_names, (list, tuple)):
            assert isinstance(grid_names, str)
            grid_names = [grid_names]
        assert len(grid_data.shape) == (self.dim + 2)
        assert len(grid_names) == grid_data.shape[1]
        grid_data_rem = []
        grid_names_rem = []
        for grid_channel, grid_name in enumerate(grid_names):
            current_grid_data = grid_data[:, [grid_channel]]
            if grid_name in names_dict['static']:
                data_channel = names_dict['static'].index(grid_name)
                current_grid_data = current_grid_data[[0]]
                assert data_dict['static'][:, [data_channel]].shape == current_grid_data.shape
                data_dict['static'][:, [data_channel]] = current_grid_data
            elif grid_name in names_dict['dynamic']:
                data_channel = names_dict['dynamic'].index(grid_name)
                assert data_dict['dynamic'][:, [data_channel]].shape == current_grid_data.shape
                data_dict['dynamic'][:, [data_channel]] = current_grid_data
            else:
                grid_data_rem.append(grid_name)
                grid_data_rem.append(current_grid_data)
        if len(grid_data_rem) > 0:
            grid_data_rem = np.concatenate(grid_data_rem, axis=1)
        else:
            grid_data_rem = None
        return grid_data_rem, grid_names_rem

    @staticmethod
    def _split_data_dicts(data_dict, grid_names):
        for key in data_dict.keys():
            data = data_dict[key]
            if data is not None:
                current_grid_names = grid_names[key]
                data_dict.update({key: None})
                data = np.split(data, len(current_grid_names), axis=1)
                data_dict.update({
                    key: dict(zip(current_grid_names, data))
                })

    @staticmethod
    def _combine_data_dicts(data_dict):
        for key in data_dict.keys():
            data = data_dict[key]
            if data is not None:
                data_dict.update({key: None})
                data = np.concatenate(list(data.values()), axis=1)
                data_dict.update({key: data})

    @staticmethod
    def _to_datetime_range(time_range):
        min_time_range = time_range[0]
        max_time_range = time_range[1]
        min_date = datetime.datetime.combine(
            datetime.date(*min_time_range, 1), datetime.time.min
        )
        max_date = datetime.datetime.combine(
            datetime.date(*max_time_range, monthrange(*max_time_range)[1]),
            datetime.time.max
        )
        return min_date, max_date