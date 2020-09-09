import torch.nn as nn
from itertools import product


class ParametricModule(nn.Module):
    __options__ = {}

    def __init__(self, **kwargs):
        super(ParametricModule, self).__init__()

        for kw in kwargs.keys():
            if kw not in self.__class__.__options__:
                raise TypeError(
                    "[ERROR] {} got an unexpected keyword argument '{}'".format(self.__class__.__name__, kw)
                )

        for option in self.__class__.__options__.keys():
            if option in kwargs:
                value = kwargs[option]
            else:
                value = self.__class__.__options__[option]
            setattr(self, option, value)

    def _require_not_none(self, *attr_names):
        for attr_name in attr_names:
            assert getattr(self, attr_name) is not None, "[ERROR] Attribute <{}> of module <{}> may not be None.".format(
                attr_name, self.__class__.__name__
            )

    @classmethod
    def from_config(cls, config, input_channels, output_channels=None):
        kwargs = {kw: config[kw] for kw in config if kw in cls.__options__}
        kwargs.update({'input_channels': input_channels})
        if output_channels is not None:
            kwargs.update({'output_channels': output_channels})
        return cls(**kwargs)

    def create_parameter_pool(self, **kwargs):
        for kw in kwargs.keys():
            if kw not in self.__class__.__options__:
                raise TypeError(
                    "[ERROR] <create_parameter_pool> got an unexpected keyword argument '{}'".format(self.__class__.__name__, kw)
                )
        param_settings = {
            option: (kwargs[option] if option in kwargs else [self.__class__.__options__[option]])
            for option in self.__class__.__options__.keys()
        }
        param_combos = list(
            product(
                *list(param_settings.values())
            )
        )
        param_combos = [
            dict(zip(self.__class__.__options__.keys(), combo)) for combo in param_combos
        ]
        print("[INFO] Number of parameter combos for network {}: {}".format(
            self.__class__.__name__, len(param_combos))
        )
        return [ParameterSetting(**combo) for combo in param_combos]


class ParameterSetting(object):
    def __init__(self, **kwargs):
        self.options = kwargs

    @classmethod
    def from_module(cls, module: ParametricModule):
        kwargs = {
            option: getattr(module, option) for option in module.__class__.__options__
        }
        return ParameterSetting(**kwargs)

    def update_config(self, config, module_type):
        config['network'][module_type].update({'options': {}})
        config['network'][module_type].update({'options': self.options})
        return config

    def to_dict(self):
        return self.options

    def __repr__(self):
        return repr(self.options)

    def __str__(self):
        return str(self.options)