from enum import Enum
import torch
import torch.nn as nn

from networks.modular_downscaling_model.base_modules import ParametricModule, ConvBlock, ConvMultiBlock, DeconvBlock
from networks.modular_downscaling_model.base_modules import ResNetBlock, ResNetMultiBlock, InceptionBlock
from networks.modular_downscaling_model.base_modules import MaxPoolBlock, ResamplingBlock2D, UpsamplingConvBlock
from networks.modular_downscaling_model.base_modules import SpaceToDepthBlock, DepthToSpaceBlock

from networks.modular_downscaling_model.input_modules import *
from networks.modular_downscaling_model.core_modules import *
from networks.modular_downscaling_model.output_modules import *


class ParametricSequential(ParametricModule):

    __options__ = {
        'input_channels': None,
        'output_channels': None,
        'num_modules': None
    }

    def __init__(self, *modules):
        assert len(modules) > 0
        assert hasattr(modules[0], 'input_channels')
        assert hasattr(modules[-1], 'output_channels')
        kwargs = {
            'input_channels': modules[0].input_channels,
            'output_channels': modules[-1].output_channels,
            'num_modules': len(modules)
        }
        super(ParametricSequential, self).__init__(**kwargs)
        self.module_sequence = nn.ModuleList(modules)

    def forward(self, *args):
        for module in self.module_sequence:
            if isinstance(args, tuple):
                args = module(*args)
            else:
                args = module(args)
        return args

    @classmethod
    def from_config(cls, config, input_channels, output_channels=None):
        assert isinstance(config, list)
        assert len(config) > 0
        model_configurator = ModelConfigurator()
        modules = []
        for i, module_config in enumerate(config):
            if i < len(config) - 1:
                current_module = model_configurator.build_module(module_config, input_channels)
            else:
                current_module = model_configurator.build_module(module_config, input_channels, output_channels)
            modules.append(current_module)
            input_channels = current_module.output_channels
        return cls(*modules)


class ConcatenationBlock(ParametricModule):
    __options__ = {
        'input_channels': None,
        'output_channels': None,
        'num_modules': None
    }

    def __init__(self, *modules):
        assert len(modules) > 0
        super(ConcatenationBlock, self).__init__(num_modules=len(modules))
        self.branches = nn.ModuleList(modules)
        self.input_channels = []
        self.output_channels = 0
        for module in modules:
            self.input_channels.append(module.input_channels)
            self.output_channels += module.output_channels

    def forward(self, *args):
        assert len(args) == self.num_modules
        output = []
        for arg, module in zip(args, self.branches):
            output.append(module(arg))
        output = torch.cat(output, dim=1)
        return output

    @classmethod
    def from_config(cls, config, input_channels, output_channels=None):
        if output_channels is not None:
            raise Exception('[ERROR] Number of output channels can not be specified arbitrarily.')
        assert 'modules' in config
        assert isinstance(config['modules'], list)
        assert len(config['modules']) > 0
        if not isinstance(input_channels, list):
            input_channels = [input_channels]
        # assert len(input_channels) == len(config['modules'])
        model_configurator = ModelConfigurator()
        modules = [
            model_configurator.build_module(module_config, channels)
            for module_config, channels in zip(config['modules'], input_channels)
        ]
        return cls(*modules)


class ModuleType(Enum):
    IN_LR_CONV = 'INPUT_LR_CONV'
    IN_LR_RES = 'INPUT_LR_RES'
    IN_LR_INC = 'INPUT_LR_INC'
    IN_LR_INT = 'INPUT_LR_INT'
    IN_HR_CONV = 'INPUT_HR_CONV'
    IN_HR_RES = 'INPUT_HR_RES'
    IN_HR_INC = 'INPUT_HR_INC'
    IN_ID = 'INPUT_ID'
    UNET_CONV = 'UNET_CONV'
    UNET_RES = 'UNET_RES'
    ENET = "ENHANCENET"
    UNET_RES_SUPER = "UNET_RES_SUPER"
    SUPER_CONV = 'SUPER_CONV'
    SUPER_RES = 'SUPER_RES'
    UP_CONV = 'UP_CONV'
    CAT = 'CAT'
    CONV = 'CONV'
    DECONV = 'DECONV'
    MCONV = 'MULTI_CONV',
    MAXPOOL = 'MAX_POOL'
    RES = 'RES'
    MRES = 'MULTI_RES',
    INC = 'INC'
    INT = 'INT'
    SKIP = 'SKIP'
    SEQ = 'SEQ'
    S2D = 'S2D'
    D2S = 'D2S'
    LOCLIN = 'LOCLIN'
    LOCANN = 'LOCANN'
    LINCNNT = 'LINCNNT'


__module_types__ = {
    ModuleType.IN_LR_CONV: ConvModuleLR,
    ModuleType.IN_LR_RES: ResNetModuleLR,
    ModuleType.IN_LR_INC: InceptionModuleLR,
    ModuleType.IN_LR_INT: InterpolationModuleLR,
    ModuleType.IN_HR_CONV: ConvModuleHR,
    ModuleType.IN_HR_RES: ResNetModuleHR,
    ModuleType.IN_HR_INC: InceptionModuleHR,
    ModuleType.IN_ID: IdentityModule,
    ModuleType.UNET_CONV: StandardUNet,
    ModuleType.UNET_RES: ResUNet,
    ModuleType.UNET_RES_SUPER: ResUNetSuper,
    ModuleType.ENET: EnhanceNet,
    ModuleType.SUPER_CONV: ConvSupersamplingModule,
    ModuleType.SUPER_RES: ResNetSupersamplingModule,
    ModuleType.UP_CONV: UpsamplingConvBlock,
    ModuleType.MAXPOOL: MaxPoolBlock,
    ModuleType.CAT: ConcatenationBlock,
    ModuleType.CONV: ConvBlock,
    ModuleType.MCONV: ConvMultiBlock,
    ModuleType.DECONV: DeconvBlock,
    ModuleType.RES: ResNetBlock,
    ModuleType.MRES: ResNetMultiBlock,
    ModuleType.INC: InceptionBlock,
    ModuleType.INT: ResamplingBlock2D,
    ModuleType.SKIP: 'SKIP', # not implemented yet
    ModuleType.SEQ: ParametricSequential,
    ModuleType.S2D: SpaceToDepthBlock,
    ModuleType.D2S: DepthToSpaceBlock,
    ModuleType.LOCLIN: LocalizedLinearModel,
    ModuleType.LOCANN: LocalizedANN,
    ModuleType.LINCNNT: LinearCNNTranspose
}


class ModelConfigurator(object):
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

    def build_model(self, config):
        input_channels = []
        input_channels_lr = len(self.input_grids_lr)
        if input_channels_lr > 0:
            input_channels.append(input_channels_lr)
        input_channels_hr = len(self.input_grids_hr)
        if input_channels_hr > 0:
            input_channels.append(input_channels_hr)
        if len(input_channels) == 1:
            input_channels = input_channels[0]
        output_channels = len(self.target_grids)
        if isinstance(config, dict):
            if 'name' in config:
                model_name = config['name']
                assert 'modules' in config
                model = self.build_module(config['modules'], input_channels, output_channels)
                model.name = model_name
            else:
                model = self.build_module(config, input_channels, output_channels)
        else:
            model = self.build_module(config, input_channels, output_channels)
        return model

    @staticmethod
    def build_module(config, input_channels, output_channels=None):
        if isinstance(config, dict):
            assert 'type' in config
            module_type = ModuleType(config['type'].upper())
            module_class = __module_types__[module_type]
        elif isinstance(config, list):
            module_class = ParametricSequential
        else:
            raise Exception('[ERROR] Unknown configuration format.')
        module = module_class.from_config(config, input_channels, output_channels)
        return module
