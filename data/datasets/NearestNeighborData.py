import numpy as np
from .LowResHighResDataset import LowResHighResDataset, region_geometry


class NearestNeighborData(LowResHighResDataset):
    def __init__(self, dataset: LowResHighResDataset, num_models=None, model_index=None, k=16):
        super(NearestNeighborData, self).__init__(
            dataset.geometry_lr, dataset.geometry_hr,
            dataset.grid_names_lr, dataset.grid_names_hr,
            dataset.grid_names_target
        )
        if isinstance(self.geometry_lr, dict):
            assert len(self.geometry_lr.keys()) == 1, '[ERROR] NearestNeighborData is not thought to be used with multi-region datasets.'
            self.geometry_lr = self.geometry_lr[list(self.geometry_lr.keys())[0]]
        if isinstance(self.geometry_hr, dict):
            assert len(self.geometry_hr.keys()) == 1, '[ERROR] NearestNeighborData is not thought to be used with multi-region datasets.'
            self.geometry_hr = self.geometry_hr[list(self.geometry_hr.keys())[0]]
        self.num_nearest_neighbors_lr = k
        self.num_nearest_neighbors_hr = 12 * k
        self._set_nearest_neighbor_indices(model_index, num_models)
        self._read_dataset(dataset)
        self._in_grid_mode = False
        self._reset_mask_hr()

    def _set_nearest_neighbor_indices(self, model_index, num_models):
        mask_lr = self.geometry_lr.mask
        lon_lr = self.geometry_lr.lon
        lat_lr = self.geometry_lr.lat
        mask_hr = self.geometry_hr.mask
        lon_hr = self.geometry_hr.lon
        lat_hr = self.geometry_hr.lat
        max_num_models = np.sum(1 - mask_hr)
        if num_models is None:
            assert model_index is not None
            assert len(model_index) <= max_num_models
            assert max(model_index) < max_num_models
        else:
            if model_index is not None:
                assert len(model_index) == num_models
                assert len(model_index) <= max_num_models
                assert max(model_index) < max_num_models
            else:
                model_index = np.arange(max_num_models).astype(int)
                if num_models < max_num_models:
                    np.random.shuffle(model_index)
                    model_index = np.sort(model_index[:num_models])
        self.num_models = len(model_index)

        valid_lon_lr = lon_lr[mask_lr == 0]
        valid_lat_lr = lat_lr[mask_lr == 0]
        valid_lon_hr = lon_hr[mask_hr == 0]
        valid_lat_hr = lat_hr[mask_hr == 0]

        index_lon_lr, index_lat_lr = np.meshgrid(np.arange(self.shape_lr[1]), np.arange(self.shape_lr[0]))
        index_lon_hr, index_lat_hr = np.meshgrid(np.arange(self.shape_hr[1]), np.arange(self.shape_hr[0]))

        valid_index_lon_lr = index_lon_lr[mask_lr == 0].astype(int)
        valid_index_lat_lr = index_lat_lr[mask_lr == 0].astype(int)
        valid_index_lon_hr = index_lon_hr[mask_hr == 0].astype(int)
        valid_index_lat_hr = index_lat_hr[mask_hr == 0].astype(int)

        self.model_index_lon = valid_index_lon_hr[model_index]
        self.model_index_lat = valid_index_lat_hr[model_index]

        input_index_lon_lr = []
        input_index_lat_lr = []
        input_index_lon_hr = []
        input_index_lat_hr = []

        for i in model_index:
            nn_dist = self._nearest_neighbor_metric(
                lon=valid_lon_lr, lat=valid_lat_lr,
                lon_0=valid_lon_hr[i], lat_0=valid_lat_hr[i]
            )
            rank_index_lr = np.argpartition(nn_dist, self.num_nearest_neighbors_lr)[:self.num_nearest_neighbors_lr]
            input_index_lon_lr.append(valid_index_lon_lr[rank_index_lr])
            input_index_lat_lr.append(valid_index_lat_lr[rank_index_lr])

        self.input_index_lon_lr = np.array(input_index_lon_lr)
        self.input_index_lat_lr = np.array(input_index_lat_lr)

        for i in model_index:
            nn_dist = self._nearest_neighbor_metric(
                lon=valid_lon_hr, lat=valid_lat_hr,
                lon_0=valid_lon_hr[i], lat_0=valid_lat_hr[i]
            )
            rank_index_hr = np.argpartition(nn_dist, self.num_nearest_neighbors_hr)[:self.num_nearest_neighbors_hr]
            input_index_lon_hr.append(valid_index_lon_hr[rank_index_hr])
            input_index_lat_hr.append(valid_index_lat_hr[rank_index_hr])

        self.input_index_lon_hr = np.array(input_index_lon_hr)
        self.input_index_lat_hr = np.array(input_index_lat_hr)

        self.num_features = (
                self.num_nearest_neighbors_lr * len(self.input_grids_lr()) +
                self.num_nearest_neighbors_hr * len(self.input_grids_hr())
        )

    @staticmethod
    def _nearest_neighbor_metric(lon=None, lat=None, lon_0=None, lat_0=None):
        return np.abs(lon - lon_0) + np.abs(lat - lat_0)

    def _read_dataset(self, dataset: LowResHighResDataset):
        self.input_lr = {'static': [], 'dynamic': []}
        if len(self.grid_names_lr['dynamic']):
            data, _ = dataset.get_input_lr(self.grid_names_lr['dynamic'])
            for idx_lat, idx_lon in zip(self.input_index_lat_lr, self.input_index_lon_lr):
                self.input_lr['dynamic'].append(np.reshape(data[:, :, idx_lat, idx_lon], newshape=(data.shape[0], -1)))
        self.input_lr.update(
            {
                'dynamic': np.stack(self.input_lr['dynamic'], axis=1)
                if len(self.input_lr['dynamic']) > 0 else None
            }
        )
        if len(self.grid_names_lr['static']):
            data, _ = dataset.get_input_lr(self.grid_names_lr['static'])
            for idx_lat, idx_lon in zip(self.input_index_lat_lr, self.input_index_lon_lr):
                self.input_lr['static'].append(np.reshape(data[:, :, idx_lat, idx_lon], newshape=(data.shape[0], -1)))
        self.input_lr.update(
            {
                'static': np.stack(self.input_lr['static'], axis=1)
                if len(self.input_lr['static']) > 0 else None
            }
        )
        self.input_hr = {'static': [], 'dynamic': []}
        if len(self.grid_names_hr['dynamic']):
            data, _ = dataset.get_input_hr(self.grid_names_hr['dynamic'])
            for idx_lat, idx_lon in zip(self.input_index_lat_hr, self.input_index_lon_hr):
                self.input_hr['dynamic'].append(np.reshape(data[:, :, idx_lat, idx_lon], newshape=(data.shape[0], -1)))
        self.input_hr.update(
            {
                'dynamic': np.stack(self.input_hr['dynamic'], axis=1)
                if len(self.input_hr['dynamic']) > 0 else None
            }
        )
        if len(self.grid_names_hr['static']):
            data, _ = dataset.get_input_hr(self.grid_names_hr['static'])
            for idx_lat, idx_lon in zip(self.input_index_lat_hr, self.input_index_lon_hr):
                self.input_hr['static'].append(np.reshape(data[:, :, idx_lat, idx_lon], newshape=(data.shape[0], -1)))
        self.input_hr.update(
            {
                'static': np.stack(self.input_hr['static'], axis=1)
                if len(self.input_hr['static']) > 0 else None
            }
        )
        self.target = {'static': [], 'dynamic': []}
        if len(self.grid_names_target['dynamic']):
            data, _ = dataset.get_target(self.grid_names_target['dynamic'])
            for idx_lat, idx_lon in zip(self.model_index_lat, self.model_index_lon):
                self.target['dynamic'].append(np.reshape(data[:, :, idx_lat, idx_lon], newshape=(data.shape[0], -1)))
        self.target.update(
            {
                'dynamic': np.stack(self.target['dynamic'], axis=1).transpose((0, 2, 1))
                if len(self.target['dynamic']) > 0 else None
            }
        )
        if len(self.grid_names_target['static']):
            data, _ = dataset.get_target(self.grid_names_target['static'])
            for idx_lat, idx_lon in zip(self.model_index_lat, self.model_index_lon):
                self.target['static'].append(np.reshape(data[:, :, idx_lat, idx_lon], newshape=(data.shape[0], -1)))
        self.target.update(
            {
                'static': np.stack(self.target['static'], axis=1).transpose((0, 2, 1))
                if len(self.target['static']) > 0 else None
            }
        )
        self._len = len(dataset)

    def _reset_mask_hr(self):
        mask_hr = np.ones_like(self.geometry_hr.mask)
        mask_hr[self.model_index_lat, self.model_index_lon] = 0
        self.geometry_hr = region_geometry(
            self.geometry_hr.lon,
            self.geometry_hr.lat,
            mask_hr
        )

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        target = [self.target['dynamic'][item]]
        if self.target['static'] is not None:
            target.append(self.target['static'][0])
        target = np.concatenate(target, axis=0)
        input_lr = [self.input_lr['dynamic'][item]]
        if self.input_lr['static'] is not None:
            input_lr.append(self.input_lr['static'][0])
        if len(input_lr) > 0:
            input_lr = np.concatenate(input_lr, axis=-1)
        else:
            input_lr = []
        input_hr = []
        if self.input_hr['dynamic'] is not None:
            input_hr.append(self.input_hr['dynamic'][item])
        if self.input_hr['static'] is not None:
            input_hr.append(self.input_hr['static'][0])
        if len(input_hr) > 0:
            input_hr = np.concatenate(input_hr, axis=-1)
        else:
            input_hr = []
        mask_lr = self.geometry_lr.mask
        mask_hr = self.geometry_hr.mask
        return target, input_lr, input_hr, mask_lr, mask_hr

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

    def grids(self):
        raise NotImplementedError()

    def samples(self):
        raise NotImplementedError()
