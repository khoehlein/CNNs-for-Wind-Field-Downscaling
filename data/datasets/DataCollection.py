import numpy as np
from .LowResHighResDataset import LowResHighResDataset
from .DataSection import DataSection
from .PatchDataSection import PatchDataSection


class DataCollection(LowResHighResDataset):
    def __init__(self, datasets):
        if not isinstance(datasets, (list, tuple)):
            datasets = [datasets]
        assert len(datasets) > 0
        base_dataset = datasets[0].samples()
        assert isinstance(base_dataset, (DataSection, PatchDataSection))
        super(DataCollection, self).__init__(
            {base_dataset.region: base_dataset.geometry_lr},
            {base_dataset.region: base_dataset.geometry_hr},
            base_dataset.grid_names_lr,
            base_dataset.grid_names_hr,
            base_dataset.grid_names_target
        )
        self.datasets = {base_dataset.region: [base_dataset]}
        self._sort_datasets(datasets[1:])
        self._set_index()
        self._in_grid_mode = False

    def __len__(self):
        return len(self._index)

    def __getitem__(self, item):
        region_index, section_index, sample_index = tuple(self._index[item])
        output = self.datasets[self._region_names[region_index]][section_index][sample_index]
        return output

    def get_input_lr(self, grid_names):
        return self._get_section_data('get_input_lr', grid_names)

    def get_input_hr(self, grid_names):
        return self._get_section_data('get_input_hr', grid_names)

    def get_target(self, grid_names):
        return self._get_section_data('get_target', grid_names)

    def set_input_lr(self, grid_names, data):
        return self._set_section_data('set_input_lr', grid_names, data)

    def set_input_hr(self, grid_names, data):
        return self._set_section_data('set_input_hr', grid_names, data)

    def set_target(self, grid_names, data):
        return self._set_section_data('set_target', grid_names, data)

    def grids(self):
        for region in self.datasets.keys():
            sections = self.datasets[region]
            for section in sections:
                section.grids()
        self._in_grid_mode = True

    def samples(self):
        for region in self.datasets.keys():
            sections = self.datasets[region]
            for section in sections:
                section.samples()
        self._in_grid_mode = False

    def _sort_datasets(self, datasets):
        for dataset in datasets:
            assert isinstance(dataset, (DataSection, PatchDataSection))
            assert self.shape_lr == dataset.shape_lr
            assert self.shape_hr == dataset.shape_hr
            assert self.grid_names_lr == dataset.grid_names_lr
            assert self.grid_names_hr == dataset.grid_names_hr
            assert self.grid_names_target == dataset.grid_names_target
            region = dataset.region
            if region in self.datasets:
                assert np.all([
                    np.all(new_data == old_data)
                    for old_data, new_data in zip(self.geometry_lr[region], dataset.geometry_lr)
                ])
                assert np.all([
                    np.all(new_data == old_data)
                    for old_data, new_data in zip(self.geometry_hr[region], dataset.geometry_hr)
                ])
                self.datasets[region].append(dataset.samples())
            else:
                self.datasets.update({region: [dataset.samples()]})
                self.geometry_lr.update({region: dataset.geometry_lr})
                self.geometry_hr.update({region: dataset.geometry_hr})

    def _set_index(self):
        index = []
        num_sections = 0
        for i, region in enumerate(self.datasets.keys()):
            region_index = []
            sections = self.datasets[region]
            for j, section in enumerate(sections):
                num_samples = len(section)
                sample_index = np.arange(num_samples, dtype=int)
                section_index = np.zeros((num_samples, 3), dtype=int)
                section_index[:, 0] = i
                section_index[:, 1] = j
                section_index[:, 2] = sample_index
                region_index.append(section_index)
            num_sections += len(sections)
            region_index = np.concatenate(region_index, axis=0)
            index.append(region_index)
        index = np.concatenate(index, axis=0)
        self._index = index
        self._region_names = np.array(list(self.datasets.keys()))
        self._num_sections = num_sections

    def _get_section_data(self, method_name, grid_names):
        output = []
        output_names = []
        for region in self._region_names:
            sections = self.datasets[region]
            for section in sections:
                method = getattr(section, method_name)
                section_data, output_names = method(grid_names)
                output.append(section_data)
        if len(output) > 0:
            output = np.concatenate(output, axis=0)
        else:
            output = None
        return output, output_names

    def _set_section_data(self, method_name, grid_names, data):
        data = self._split_data(data)
        i = 0
        data_rem = []
        names_rem = []
        for region in self._region_names:
            sections = self.datasets[region]
            for section in sections:
                method = getattr(section, method_name)
                section_rem, names_rem = method(grid_names, data[i])
                data_rem.append(section_rem)
                i += 1
        if len(data_rem) > 0:
            if data_rem[0] is not None:
                data_rem = np.concatenate(data_rem, axis=0)
            else:
                data_rem = None
        else:
            data_rem = None
        return data_rem, names_rem

    def _split_data(self, data):
        shape = data.shape
        if shape[0] == len(self):
            split_index = np.argwhere(self._index[:, -1] == 0).flatten()
            if len(split_index) > 1:
                data = np.split(data, split_index[1:], axis=0)
            else:
                data = (data,)
        elif shape[0] == self._num_sections:
            data = np.array_split(data, self._num_sections, axis=0)
        else:
            raise Exception('[ERROR] Unknown data shape.')
        return data
