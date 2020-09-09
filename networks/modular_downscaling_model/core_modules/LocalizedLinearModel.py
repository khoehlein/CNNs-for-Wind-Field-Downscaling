import numpy as np
import torch
import torch.nn as nn
from data.datasets.LowResHighResDataset import region_geometry
from networks.modular_downscaling_model.base_modules import ParametricModule


class LocalizedLinearModel(ParametricModule):

    __options__ = {
        "input_channels": None,
        "output_channels": None,
        "num_models": 30000,
        "num_nearest_neighbors_lr": 16,
        "num_nearest_neighbors_hr": None
    }

    def __init__(self, **kwargs):
        super(LocalizedLinearModel, self).__init__(**kwargs)
        self._require_not_none('input_channels', 'output_channels')
        self.shape_lr = None
        self.shape_hr = None
        self.input_index_lon_lr = None
        self.input_index_lat_lr = None
        self.input_index_lon_hr = None
        self.input_index_lat_hr = None
        self.model_index_lon = None
        self.model_index_lat = None
        self.model_index = None
        if self.num_nearest_neighbors_hr is None:
            self.num_nearest_neighbors_hr = 12 * self.num_nearest_neighbors_lr
        self.model = None

    def set_model_index(self, geometry_lr, geometry_hr, model_index=None):
        assert isinstance(geometry_lr, (region_geometry, dict))
        assert isinstance(geometry_hr, (region_geometry, dict))
        if isinstance(geometry_lr, dict):
            geometry_lr = geometry_lr[list(geometry_lr.keys())[0]]
        self.shape_lr = geometry_lr[0].shape
        if isinstance(geometry_hr, dict):
            geometry_hr = geometry_hr[list(geometry_hr.keys())[0]]
        self.shape_hr = geometry_hr.mask.shape
        index_lon_hr, index_lat_hr = np.meshgrid(np.arange(self.shape_hr[1]), np.arange(self.shape_hr[0]))
        valid_index_lon_hr = index_lon_hr[geometry_hr.mask == 0].astype(int)
        valid_index_lat_hr = index_lat_hr[geometry_hr.mask == 0].astype(int)
        max_num_models = np.sum(1 - geometry_hr.mask)
        if model_index is None:
            model_index = np.arange(max_num_models).astype(int)
            if self.num_models < len(model_index):
                np.random.shuffle(model_index)
                model_index = np.sort(model_index[:self.num_models])
        else:
            assert len(model_index) <= max_num_models
            assert max(model_index) <= max_num_models
        self.model_index = model_index
        self.num_models = len(model_index)
        self.model_index_lon = valid_index_lon_hr[model_index]
        self.model_index_lat = valid_index_lat_hr[model_index]
        if isinstance(self.input_channels, list):
            num_features = self.num_nearest_neighbors_lr * self.input_channels[0]
            num_features += self.num_nearest_neighbors_hr * self.input_channels[1]
        else:
            num_features = self.num_nearest_neighbors_lr * self.input_channels
        self.model = nn.Conv1d(
            in_channels=self.num_models,
            out_channels=self.output_channels * self.num_models,
            groups=self.num_models,
            kernel_size=num_features
        )

    def forward(self, x):
        if self.model is None:
            raise Exception("[ERROR] Need to set geometry before applying the model.")
        output = self.model(x)
        return self._reshape_output(output)

    def _reshape_output(self, model_output):
        batch_size = model_output.size(0)
        # output = torch.zeros(batch_size, self.output_channels, *self.shape_hr, device=model_output.device)
        model_output = torch.cat(torch.split(model_output, self.output_channels, dim=1), dim=2)
        # output[:, :, self.model_index_lat, self.model_index_lon] = model_output
        return model_output
