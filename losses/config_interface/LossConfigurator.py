import numpy as np
from losses.loss_functions import *
from losses.config_interface import LossType,  LossCollection


__loss_functions__ = {
    LossType.L1: L1,
    LossType.MAE: MAE,
    LossType.MSE: MSE,
    LossType.COSSIM: CosineSimilarity,
    LossType.COSDIS: CosineDissimilarity,
    # LossType.WEIGHTED: NotImplementedError()
}


class LossConfigurator(object):
    def __init__(self):
        pass

    def build_losses(self, config):
        step_names = ["training", "validation", "test"]
        if isinstance(config, dict):
            step_separation = np.any([(step_name in config) for step_name in step_names])
            if step_separation:
                losses = {}
                global_options = self._read_options(config)
                for step_name in step_names:
                    if step_name in config:
                        step_config = config[step_name]
                        losses.update({step_name: self._read_loss_config(step_config, global_options)})
                # append objective configuration to other steps
                if "training" in losses:
                    if len(losses["training"]) > 0:
                        objective = losses["training"][0]
                        for step_name in losses.keys():
                            if step_name != "training":
                                step_losses = losses[step_name]
                                loss_names = [loss.name for loss in step_losses]
                                if objective.name not in loss_names:
                                    step_losses.append(objective)
            else:
                losses = {"training": [self._read_loss_config(config)]}
        elif isinstance(config, (list, tuple)):
            losses = {"training": self._read_loss_config(config)}
        else:
            raise Exception('[ERROR] Unknown configuration format.')
        for step_name in step_names:
            if step_name in losses:
                losses.update({
                    step_name: LossCollection(*losses[step_name])
                })
        return losses

    def _read_loss_config(self, config, global_options=None):
        loss_list = []
        if isinstance(config, dict):
            if "losses" in config:
                local_options = self._read_options(config, global_options)
                loss_config = config["losses"]
                loss_list = self._read_loss_config(loss_config, local_options)
            if "type" in config:
                loss_type = LossType(config["type"])
                local_options = self._read_options(config, global_options)
                kwargs = {
                    kw: config[kw]
                    for kw in config.keys()
                    if kw not in local_options and kw != "losses"
                }
                loss_constructor = __loss_functions__[loss_type]
                loss_config = global_options.copy() if global_options is not None else {}
                loss_config.update(config)
                if loss_type != LossType.WEIGHTED:
                    loss_list = [loss_constructor(**{**loss_config, **kwargs})]
                else:
                    raise NotImplementedError()
                    # assert len(loss_list) > 0
                    # loss_list = [loss_constructor(*loss_list, **{**loss_config, **kwargs})]
        elif isinstance(config, (list, tuple)):
            for loss_config in config:
                loss_list += self._read_loss_config(loss_config, global_options)
        elif isinstance(config, str):
            loss_type = LossType(config.upper())
            assert loss_type != LossType.WEIGHTED
            loss_constructor = __loss_functions__[loss_type]
            kwargs = global_options.copy() if global_options is not None else {}
            loss_list = [loss_constructor(**kwargs)]
        else:
            raise Exception('[ERROR] Unknown configuration format.')
        return loss_list

    @staticmethod
    def _read_options(config, global_options=None):
        option_names = ['use_scalings', 'use_mask', 'batch_reduction', 'spatial_reduction']
        if global_options is None:
            option_dict = {kw: None for kw in option_names}
        else:
            option_dict = {kw: global_options[kw] for kw in option_names}
        for kw in option_names:
            if kw in config:
                option_dict.update({kw: config[kw]})
        return option_dict
