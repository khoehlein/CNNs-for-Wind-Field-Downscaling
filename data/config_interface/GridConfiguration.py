from data.datasets import LowResHighResDataset


class GridConfiguration(object):
    def __init__(
            self,
            input_grids_lr=None, input_grids_hr=None, target_grids=None,
            input_scalings_lr=None, input_scalings_hr=None, target_scalings=None
    ):
        self.input_grids_lr = input_grids_lr
        self.input_grids_hr = input_grids_hr
        self.target_grids = target_grids
        self.target_channels_in_lr = [
            self.input_grids_lr.index(grid) if grid in self.input_grids_lr else None
            for grid in self.target_grids
        ]
        self.input_scalings_lr = input_scalings_lr
        self.input_scalings_hr = input_scalings_hr
        self.target_scalings = target_scalings

    def to(self, device, dtype=None):
        for grid_scaling in self.input_scalings_lr:
            if grid_scaling.scaler is not None:
                grid_scaling.scaler.to(device, dtype=dtype)
        for grid_scaling in self.input_scalings_hr:
            if grid_scaling.scaler is not None:
                grid_scaling.scaler.to(device, dtype=dtype)
        for grid_scaling in self.target_scalings:
            if grid_scaling.scaler is not None:
                grid_scaling.scaler.to(device, dtype=dtype)
        return self

    def fit(self, dataset):
        print("[INFO] Fitting data scalers.")
        assert isinstance(dataset, LowResHighResDataset)
        assert dataset.input_grids_lr() == self.input_grids_lr
        assert dataset.input_grids_hr() == self.input_grids_hr
        assert dataset.target_grids() == self.target_grids
        for grid_scaling in self.input_scalings_lr:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_input_lr(grid_names)[0]
                scaler.fit(data)
        for grid_scaling in self.input_scalings_hr:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_input_hr(grid_names)[0]
                scaler.fit(data)
        for grid_scaling in self.target_scalings:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_target(grid_names)[0]
                scaler.fit(data)
        return self.input_scalings_lr, self.input_scalings_hr, self.target_scalings

    def transform(self, dataset):
        print("[INFO] Transforming dataset.")
        assert isinstance(dataset, LowResHighResDataset)
        assert dataset.input_grids_lr() == self.input_grids_lr
        assert dataset.input_grids_hr() == self.input_grids_hr
        assert dataset.target_grids() == self.target_grids
        for grid_scaling in self.input_scalings_lr:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_input_lr(grid_names)[0]
                data = scaler.transform(data)
                dataset.set_input_lr(grid_names, data)
        for grid_scaling in self.input_scalings_hr:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_input_hr(grid_names)[0]
                data = scaler.transform(data)
                dataset.set_input_hr(grid_names, data)
        for grid_scaling in self.target_scalings:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_target(grid_names)[0]
                data = scaler.transform(data)
                dataset.set_target(grid_names, data)
        return (self.input_scalings_lr, self.input_scalings_hr, self.target_scalings), dataset

    def fit_transform(self, dataset):
        print("[INFO] Fitting data scalers and transforming datset.")
        assert isinstance(dataset, LowResHighResDataset)
        assert dataset.input_grids_lr() == self.input_grids_lr
        assert dataset.input_grids_hr() == self.input_grids_hr
        assert dataset.target_grids() == self.target_grids
        for grid_scaling in self.input_scalings_lr:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_input_lr(grid_names)[0]
                scaler.fit(data)
                data = scaler.transform(data)
                dataset.set_input_lr(grid_names, data)
        for grid_scaling in self.input_scalings_hr:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_input_hr(grid_names)[0]
                scaler.fit(data)
                data = scaler.transform(data)
                dataset.set_input_hr(grid_names, data)
        for grid_scaling in self.target_scalings:
            grid_names = grid_scaling.grids
            scaler = grid_scaling.scaler
            if scaler is not None:
                data = dataset.get_target(grid_names)[0]
                scaler.fit(data)
                data = scaler.transform(data)
                dataset.set_target(grid_names, data)
        return (self.input_scalings_lr, self.input_scalings_hr, self.target_scalings), dataset
