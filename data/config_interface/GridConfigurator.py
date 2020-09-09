from .GridConfiguration import GridConfiguration
from data.scaling import ScalerType, StandardScaler, RangeScaler
from collections import namedtuple


grid_scaling = namedtuple("grid_scaling", ["grids", "scaler"])

__scaler_types__ = {
    ScalerType.STANDARD: StandardScaler,
    ScalerType.RANGE: RangeScaler
}


class GridConfigurator(object):
    def __init__(self):
        pass

    def build_grids(self, config):
        assert isinstance(config, dict)
        assert "input_lr" in config or "input_hr" in config
        assert "target" in config
        input_grids_lr = []
        input_scalings_lr = []
        if "input_lr" in config:
            grid_list = config["input_lr"]
            input_grids_lr, input_scalings_lr = self._read_grid_list(grid_list)
        input_grids_hr = []
        input_scalings_hr = []
        if "input_hr" in config:
            grid_list = config["input_hr"]
            input_grids_hr, input_scalings_hr = self._read_grid_list(grid_list)
        grid_list = config["target"]
        target_grids, target_scalings = self._read_grid_list(grid_list)
        return GridConfiguration(
            input_grids_lr, input_grids_hr, target_grids,
            input_scalings_lr, input_scalings_hr, target_scalings
        )

    def _read_grid_list(self, grid_list):
        grid_names = []
        scalings = []
        if grid_list is None:
            return grid_names, scalings
        if not isinstance(grid_list, (list, tuple)):
            assert isinstance(grid_list, str)
            grid_list = [grid_list]
        for grid_config in grid_list:
            if isinstance(grid_config, str):
                current_grid_names = [grid_config]
                scaler = None
            elif isinstance(grid_config, list):
                if len(grid_config) == 0:
                    continue
                current_grid_names = grid_config[0]
                if current_grid_names is None:
                    continue
                if not isinstance(current_grid_names, list):
                    assert isinstance(current_grid_names, str)
                    current_grid_names = [current_grid_names]
                scaler = None
                if len(grid_config) > 1:
                    scaler_config = grid_config[1]
                    if scaler_config is not None:
                        channels = len(current_grid_names)
                        scaler = self._build_scaler(channels, grid_config[1])
                if len(grid_config) > 2:
                    raise Exception("[ERROR] Unknown configuration format.")
            else:
                raise Exception("[ERROR] Unknown configuration format.")
            grid_names += current_grid_names
            scalings.append(grid_scaling(current_grid_names, scaler))
        return grid_names, scalings

    def _build_scaler(self, channels, scaler_config):
        if scaler_config is None:
            return None
        if isinstance(scaler_config, str):
            scaler_config = [scaler_config]
        if isinstance(scaler_config, (list, tuple)):
            if len(scaler_config) == 0:
                return None
            scaler_type = ScalerType(scaler_config[0].upper())
            kwargs = {}
            if len(scaler_config) > 1:
                scaler_opts = scaler_config[1]
                if scaler_opts is None:
                    scaler_opts = {}
                assert isinstance(scaler_opts, dict)
                kwargs.update(scaler_opts)
            if len(scaler_config) > 2:
                raise Exception("[ERROR] Unknown configuration format.")
        elif isinstance(scaler_config, dict):
            if len(scaler_config) == 0:
                return None
            assert "type" in scaler_config
            scaler_type = ScalerType(scaler_config["type"].upper())
            kwargs = {}
            if "options" in scaler_config:
                scaler_opts = scaler_config["options"]
                assert isinstance(scaler_opts, dict)
                kwargs.update(scaler_opts)
        else:
            raise Exception("[ERROR] Unknown configuration format.")
        scaler_constructor = __scaler_types__[scaler_type]
        return scaler_constructor(channels=channels, **kwargs)