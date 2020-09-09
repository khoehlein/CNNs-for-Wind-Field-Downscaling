import numpy as np
import datetime
from calendar import monthrange


class DataSectionConfiguration(object):
    def __init__(
            self,
            time_range=None,
            region=None,
            directory=None,
            patching=None,
            patch_size=None,
            min_date=None,
            max_date=None,
            lr_filter_name=None,
            hr_filter_name=None
    ):
        self.min_date = None
        self.max_date = None
        if min_date is not None:
            assert max_date is not None
            self.min_date = min_date
            self.max_date = max_date
        else:
            assert time_range is not None
            self.min_date, self.max_date = self._to_datetime_range(time_range)
        assert self.min_date < self.max_date
        self.region = region
        assert region is not None
        self.directory = directory
        assert directory is not None
        self.lr_filter_name=lr_filter_name
        assert self.lr_filter_name is not None
        self.hr_filter_name=hr_filter_name
        assert self.hr_filter_name is not None
        self.patching = patching
        assert patching is not None
        self.patch_size = patch_size
        if self.patching:
            assert patch_size is not None

    def __gt__(self, other):
        if self.min_date != other.min_date:
            c = self.min_date > other.min_date
        else:
            c = self.max_date > other.max_date
        return c

    def __ge__(self, other):
        if self.min_date != other.min_date:
            c = self.min_date > other.min_date
        else:
            c = self.max_date >= other.max_date
        return c

    def __lt__(self, other):
        if self.min_date != other.min_date:
            c = self.min_date < other.min_date
        else:
            c = self.max_date < other.max_date
        return c

    def __le__(self, other):
        if self.min_date != other.min_date:
            c = self.min_date < other.min_date
        else:
            c = self.max_date <= other.max_date
        return c

    def __eq__(self, other):
        return (self.min_date == other.min_date) and (self.max_date == other.max_date)

    def __ne__(self, other):
        return (self.min_date != other.min_date) or (self.max_date != other.max_date)

    def contains(self, other):
        return (self.min_date <= other.min_date) and (self.max_date >= other.max_date)

    def overlaps(self, other):
        return (self.min_date <= other.max_date) and (self.max_date >= other.min_date)

    def touches(self, other):
        c = [
            self.min_date == self._next_higher_min_date(other.max_date),
            self.max_date == self._next_lower_max_date(other.min_date)
        ]
        return np.any(c)

    def importer_consistent_with(self, other):
        c = [
            self.directory == other.directory,
            self.lr_filter_name == other.lr_filter_name,
            self.hr_filter_name == other.hr_filter_name
        ]
        return np.all(c)

    def data_consistent_with(self, other):
        c = self.patching == other.patching
        if self.patching:
            c = c and (self.patch_size == other.patch_size)
        return c

    def region_consistent_with(self, other):
        return self.region == other.region

    def combined_with(self, other):
        assert self.overlaps(other) or self.touches(other)
        assert self.data_consistent_with(other)
        assert self.region_consistent_with(other)
        assert self.importer_consistent_with(other)
        min_date = min(self.min_date, other.min_date)
        max_date = max(self.max_date, other.max_date)
        return DataSectionConfiguration(
            time_range=None,
            region=self.region,
            directory=self.directory,
            lr_filter_name=self.lr_filter_name,
            hr_filter_name=self.hr_filter_name,
            patching=self.patching,
            patch_size=self.patch_size,
            min_date=min_date,
            max_date=max_date
        )

    def complement(self, other):
        sections = []
        if self.min_date < other.min_date:
            sections.append(
                DataSectionConfiguration(
                    time_range=None,
                    region=self.region,
                    directory=self.directory,
                    lr_filter_name=self.lr_filter_name,
                    hr_filter_name=self.hr_filter_name,
                    patching=self.patching,
                    patch_size=self.patch_size,
                    min_date=self.min_date,
                    max_date=min(
                        self.max_date,
                        self._next_lower_max_date(other.min_date)
                    )
                )
            )
        if self.max_date > other.max_date:
            sections.append(
                DataSectionConfiguration(
                    time_range=None,
                    region=self.region,
                    directory=self.directory,
                    lr_filter_name=self.lr_filter_name,
                    hr_filter_name=self.hr_filter_name,
                    patching=self.patching,
                    patch_size=self.patch_size,
                    min_date=max(
                        self.min_date,
                        self._next_higher_min_date(other.max_date),
                    ),
                    max_date=self.max_date
                )
            )
        return sections

    def copy(self):
        return DataSectionConfiguration(
            time_range=None,
            region=self.region,
            directory=self.directory,
            lr_filter_name=self.lr_filter_name,
            hr_filter_name=self.hr_filter_name,
            patching=self.patching,
            patch_size=self.patch_size,
            min_date=self.min_date,
            max_date=self.max_date
        )

    def time_range(self):
        return [
            self._to_list_date(self.min_date),
            self._to_list_date(self.max_date)
        ]

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

    @staticmethod
    def _next_lower_max_date(min_date):
        if min_date.month == 1:
            max_date = [min_date.year - 1, 12]
        else:
            max_date = [min_date.year, min_date.month - 1]
        max_date = datetime.datetime.combine(
            datetime.date(*max_date, monthrange(*max_date)[1]),
            datetime.time.max
        )
        return max_date

    @staticmethod
    def _next_higher_min_date(max_date):
        if max_date.month == 12:
            min_date = [max_date.year + 1, 1]
        else:
            min_date = [max_date.year, max_date.month + 1]
        min_date = datetime.datetime.combine(
            datetime.date(*min_date, 1), datetime.time.min
        )
        return min_date

    @staticmethod
    def _to_list_date(datetime_date):
        return [datetime_date.year, datetime_date.month]
