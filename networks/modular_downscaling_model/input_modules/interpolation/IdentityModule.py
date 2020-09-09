from networks.modular_downscaling_model.base_modules import ParametricModule


class IdentityModule(ParametricModule):
    __options__ = {
        'input_channels': None,
        'output_channels': None
    }

    def __init__(self, **kwargs):
        super(IdentityModule, self).__init__(**kwargs)
        self._require_not_none('input_channels')
        self.output_channels = self.input_channels

    def forward(self, x):
        assert x.size(1) == self.input_channels
        return x