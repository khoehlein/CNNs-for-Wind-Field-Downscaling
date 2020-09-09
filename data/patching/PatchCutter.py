import numpy as np
import numbers


class PatchCutter(object):
    def __init__(self, patch_size=None, dim=2):
        if isinstance(patch_size, numbers.Number):
            patch_size = [patch_size] * dim
        else:
            if patch_size is not None:
                assert len(patch_size) == dim
                patch_size = np.array(patch_size).astype(int)
        self.dim = dim
        self.patch_size = patch_size
        self.relative_offset = np.zeros(dim)

    def __call__(self, input):
        if self.patch_size is not None:
            if len(input) == 0:
                return input, np.zeros(2, dtype=int)
            input_dim = len(input.shape)
            assert input_dim >= self.dim
            input_size = np.array(input.shape[-self.dim:]).astype(int)
            max_shift = input_size - self.patch_size
            assert np.all(max_shift >= 0)
            lower_bounds = np.round(max_shift * self.relative_offset).astype(int)
            upper_bounds = lower_bounds + self.patch_size
            selection = (slice(None, None, None),) * (input_dim - self.dim)
            selection += tuple(slice(lb, ub, None) for lb, ub in zip(lower_bounds, upper_bounds))
            return input[selection], lower_bounds
        else:
            return input, np.zeros(2, dtype=int)

    def randomize(self):
        self.relative_offset = np.random.uniform(0, 1, 2)
        return self.relative_offset

    def synchronize(self, patch_cutter):
        self.relative_offset = patch_cutter.relative_offset
        return self.relative_offset


if __name__ == '__main__':
    patcher_a = PatchCutter(patch_size=(24, 36))
    patcher_b = PatchCutter(patch_size=(96, 108))
    a = np.random.randn(60)
    b = np.random.randn(4, 144, 180)
    patcher_a.randomize()
    patcher_b.synchronize(patcher_a)
    print(patcher_a(a).shape, patcher_b(b).shape)