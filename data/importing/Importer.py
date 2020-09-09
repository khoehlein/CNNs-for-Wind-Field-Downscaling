class Importer:
    def __init__(self, directory):
        self.directory = directory
        self.preloaded_files_low_res = []
        self.preloaded_files_high_res = []

        self._load_files()

    def _load_files(self):
        raise NotImplementedError()
