import os
import argparse
import json
import torch

from data import GridConfigurator, DataConfigurator
from networks.modular_downscaling_model import ModelConfigurator
from losses import LossConfigurator
from training.modular_downscaling_model import TrainingConfigurator

print("[INFO] Running TrainingScript.py in directory <{}>".format(os.getcwd()))

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, help="absolute or relative path to config file")
parser.add_argument('gpu', type=int, help='ID of CUDA device to use')
opt = parser.parse_args()

assert opt.config_path.endswith('.json')

with open(opt.config_path, encoding='utf-8') as f:
    config = json.load(f)

config["training"].update({"gpu": opt.gpu})

torch.set_num_threads(4)

grids = GridConfigurator().build_grids(config['grids'])

datasets = DataConfigurator(grids=grids).build_dataset(config['data'])

for step_name in ['training', 'validation', 'test']:
    if step_name in datasets:
        step_data = datasets['training']
        if step_name == 'training':
            grids.fit_transform(datasets['training'])
        else:
            grids.transform(datasets['validation'])

model = ModelConfigurator(grids=grids).build_model(config['model'])

print(model)

losses = LossConfigurator().build_losses(config['losses'])

training_process = TrainingConfigurator().build_training_process(
    grids,
    datasets,
    losses,
    model,
    config['training']
)

training_process.save_config(config)

training_process.run()

