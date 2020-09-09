import numpy as np
import os
import glob
import re

from data.importing.Importer import Importer


class NPZImporter(Importer):
    def __init__(self, directory, regions):
        self.trainingRegions = regions
        super(NPZImporter, self).__init__(directory)

        self._load_files()

    def _load_files(self):
        # for now we assume we cover the same area / region / domain
        allFiles = sorted(glob.glob(self.directory + "/**/*.npz", recursive=True))

        if len(allFiles) == 0:
            raise Exception("Dataset directory <{}> is empty.".format(self.directory))

        # divide into input and target files (ERA5 = low-res, HRES = high-res)
        allInputFiles = list(filter(lambda x: os.path.basename(x).startswith('ERA5'), allFiles))
        allTargetFiles = list(filter(lambda x: os.path.basename(x).startswith('HRES'), allFiles))

        assert (len(allInputFiles) == len(allTargetFiles))

        if len(self.preloaded_files_input) == 0:
            for filename in allInputFiles:
                curArea = re.search(r"(area[0-9])", filename).group(0)

                if curArea not in self.trainingRegions:
                    continue

                file = np.load(filename)
                print('[INFO]: Load data from file <{}>'.format(filename))

                fileContent = {}
                for key in file.files:
                    fileContent[key] = file[key]

                self.preloaded_files_input += [(fileContent, curArea)]
                file.close()

        if len(self.preloaded_files_target) == 0:
            for filename in allTargetFiles:
                curArea = re.search(r"(area[0-9])", filename).group(0)

                if curArea not in self.trainingRegions:
                    continue

                file = np.load(filename)
                print('[INFO]: Load data from file <{}>'.format(filename))

                fileContent = {}
                for key in file.files:
                    fileContent[key] = file[key]

                self.preloaded_files_target += [(fileContent, curArea)]
                file.close()