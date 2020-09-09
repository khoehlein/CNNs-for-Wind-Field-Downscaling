from .PatchData import PatchData
from . DataSection import DataSection


class PatchDataSection(PatchData):
    def __init__(self, dataset, patch_size_lr=None):
        assert isinstance(dataset, DataSection)
        super(PatchDataSection, self).__init__(dataset, patch_size_lr)
        self.region = dataset.region
        self.time_range = dataset.time_range