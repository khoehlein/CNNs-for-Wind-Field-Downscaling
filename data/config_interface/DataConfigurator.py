import numpy as np
from itertools import chain
from .DataSectionConfiguration import DataSectionConfiguration
from data.importing import HDF5Importer
from data.datasets import DataSection, DataCollection, PatchDataSection


class DataConfigurator(object):
    def __init__(
            self,
            input_grids_lr=None, input_grids_hr=None, target_grids=None,
            grids=None
    ):
        if grids is None:
            self.input_grids_lr = input_grids_lr
            self.input_grids_hr = input_grids_hr
            self.target_grids = target_grids
        else:
            self.input_grids_lr = grids.input_grids_lr
            self.input_grids_hr = grids.input_grids_hr
            self.target_grids = grids.target_grids
        self._importers = {}

    def build_dataset(self, config):
        datasets = {}
        data_sections = self.read_data_config(config)
        for step_name in data_sections.keys():
            if len(data_sections[step_name]) > 0:
                datasets.update({
                    step_name: self._build_data_from_sections(data_sections[step_name])
                })
        return datasets

    def read_data_config(self, config):
        step_names = ["training", "validation", "test"]
        if isinstance(config, dict):
            step_separation = np.any([(step_name in config) for step_name in step_names])
            if step_separation:
                global_options = self._read_options(config)
                data_sections = {}
                for step_name in step_names:
                    if step_name in config:
                        step_config = config[step_name]
                        current_data_sections = self._read_step_config(step_config, global_options)
                        self._verify_data_consistency(current_data_sections)
                        data_sections.update({
                            step_name: current_data_sections
                        })
                data_sections = self._split_simplify_sections(data_sections)
                data_sections = self._remove_section_overlap(data_sections)
            else:
                data_sections = {"training": [self._read_section_config(config)]}
        elif isinstance(config, (list, tuple)):
            data_sections = {"training": self._read_step_config(config)}
        else:
            raise Exception('[ERROR] Unknown configuration format.')
        return data_sections

    def _read_step_config(self, config, global_options=None):
        data_sections = []
        if config is None or config == [] or config == {}:
            return []
        if isinstance(config, dict):
            step_options = self._read_options(config, global_options)
            if "sections" in config:
                data_sections = self._read_step_config(config["sections"], step_options)
            else:
                data_sections.append(self._read_section_config(config, global_options))
        elif isinstance(config, (tuple, list)):
            for section_config in config:
                data_sections.append(self._read_section_config(section_config, global_options))
        else:
            raise Exception('[ERROR] Unknown configuration format.')
        return data_sections

    def _read_section_config(self, config, global_options=None):
        assert isinstance(config, dict)
        options = self._read_options(config, global_options)
        return DataSectionConfiguration(**options)

    @staticmethod
    def _verify_data_consistency(data_sections):
        if not isinstance(data_sections, (list, tuple)):
            assert isinstance(data_sections, DataSectionConfiguration)
            data_sections = [data_sections]
        c = True
        if len(data_sections) > 1:
            base_section = data_sections[0]
            for section in data_sections:
                assert isinstance(section, DataSectionConfiguration)
                if not base_section.data_consistent_with(section):
                    c = False
        assert c

    def _split_simplify_sections(self, data_sections):
        for step_name in data_sections.keys():
            sections_by_region = self._group_sections_by_region(data_sections[step_name])
            for region in sections_by_region.keys():
                sections_by_region.update({
                    region: self._simplify_section_list(sections_by_region[region])
                })
            data_sections.update({
                step_name: sections_by_region
            })
        return data_sections

    @staticmethod
    def _group_sections_by_region(data_sections):
        if not isinstance(data_sections, (list, tuple)):
            assert isinstance(data_sections, DataSectionConfiguration)
            data_sections = [data_sections]
        sections_by_region = {}
        for data_section in data_sections:
            assert isinstance(data_section, DataSectionConfiguration)
            region = data_section.region
            if data_section.region in sections_by_region:
                sections_by_region[region].append(data_section)
            else:
                sections_by_region.update({
                    region: [data_section]
                })
        return sections_by_region

    @staticmethod
    def _simplify_section_list(data_sections):
        if not isinstance(data_sections, (list, tuple)):
            assert isinstance(data_sections, DataSectionConfiguration)
            data_sections = [data_sections]
        if len(data_sections) in [0, 1]:
            return data_sections
        data_sections = sorted(data_sections)
        current_section = None
        simplified_list = []
        for new_section in data_sections:
            assert isinstance(new_section, DataSectionConfiguration)
            if current_section is None:
                current_section = new_section
            elif current_section.contains(new_section):
                pass
            elif current_section.overlaps(new_section) or current_section.touches(new_section):
                if current_section.importer_consistent_with(new_section):
                    current_section = current_section.combined_with(new_section)
                else:
                    simplified_list.append(current_section)
                    current_section = new_section.complement(current_section)[-1]
            else:
                simplified_list.append(current_section)
                current_section = new_section
        simplified_list.append(current_section)
        return simplified_list

    def _remove_section_overlap(self, data_sections):
        ranked_steps = ["test", "validation", "training"]
        blocked_sections = None
        for step_name in ranked_steps:
            if step_name in data_sections:
                sections_by_region = data_sections[step_name]
                if blocked_sections is None:
                    blocked_sections = {
                        region: self._extend_blocked_list([], sections_by_region[region])
                        for region in sections_by_region.keys()
                    }
                else:
                    for region in sections_by_region.keys():
                        section_list = sections_by_region[region]
                        if region in blocked_sections:
                            blocked_list = blocked_sections[region]
                            section_list = self._apply_blocking(blocked_list, section_list)
                            sections_by_region.update({
                                region: section_list
                            })
                            blocked_list = self._extend_blocked_list(blocked_list, section_list)
                            blocked_sections.update({
                                region: blocked_list
                            })
                        else:
                            blocked_sections.update({
                                region: self._extend_blocked_list([], section_list)
                            })
                    data_sections.update({
                        step_name: sections_by_region
                    })
        return data_sections

    def _apply_blocking(self, blocked_list, section_list):
        reduced_list = self._simplify_section_list(section_list)
        for block in blocked_list:
            assert isinstance(block, DataSectionConfiguration)
            complement_list = [section.complement(block) for section in reduced_list]
            reduced_list = list(chain.from_iterable(complement_list))
        return reduced_list

    def _extend_blocked_list(self, blocked_list, section_list):
        for section in section_list:
            block = section.copy()
            block.patching = False
            block.patch_size = None
            blocked_list.append(block)
        return self._simplify_section_list(blocked_list)

    def _build_data_from_sections(self, section_configs):
        if isinstance(section_configs, dict):
            section_configs = list(chain.from_iterable(section_configs.values()))
        if not isinstance(section_configs, (list, tuple)):
            assert isinstance(section_configs, DataSectionConfiguration)
            section_configs = [section_configs]
        datasets = []
        for section_config in section_configs:
            assert isinstance(section_config, DataSectionConfiguration)
            directory = section_config.directory
            lr_filter_name = section_config.lr_filter_name
            hr_filter_name = section_config.hr_filter_name
            if directory in self._importers:
                importer = self._importers[directory]
            else:
                importer = HDF5Importer(
                    directory,
                    low_res_filter_name=lr_filter_name,
                    high_res_filter_name=hr_filter_name
                )
                self._importers.update({
                    directory: importer
                })
            current_dataset = DataSection(
                importer,
                section_config.region, section_config.time_range(),
                input_grids_lr=self.input_grids_lr, input_grids_hr=self.input_grids_hr,
                target_grids=self.target_grids
            )
            if section_config.patching:
                current_dataset = PatchDataSection(current_dataset, patch_size_lr=section_config.patch_size)
            datasets.append(current_dataset)
        if len(datasets) == 0:
            return None
        elif len(datasets) == 1:
            return datasets[0]
        else:
            return DataCollection(datasets)

    @staticmethod
    def _read_options(config, global_options=None):
        option_names = ['region', 'time_range', 'directory', 'patching', 'patch_size', 'lr_filter_name', 'hr_filter_name']
        if global_options is None:
            option_dict = {kw: None for kw in option_names}
        else:
            option_dict = {kw: global_options[kw] for kw in option_names}
        for kw in option_names:
            if kw in config:
                option_dict.update({kw: config[kw]})
        return option_dict
