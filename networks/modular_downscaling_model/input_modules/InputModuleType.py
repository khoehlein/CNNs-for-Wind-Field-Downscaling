from enum import Enum


class InputModuleType(Enum):
    CONV = 'CONV'
    RESIDUAL = 'RESIDUAL'
    INCEPTION = 'INCEPTION'
    INTERPOLATION = 'INTERPOLATION'
