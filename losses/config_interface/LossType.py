from enum import Enum


class LossType(Enum):
    L1 = 'L1'
    MAE = 'MAE'
    MSE = 'MSE'
    COSSIM = 'COS-S'
    COSDIS = 'COS-D'
    WEIGHTED = 'WEIGHTED'