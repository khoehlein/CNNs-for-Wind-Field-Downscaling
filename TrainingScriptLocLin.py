import numpy as np
import os
import argparse
import json
import torch

# import matplotlib.pyplot as plt

from data import GridConfigurator, DataConfigurator
from data.datasets import NearestNeighborData
from networks.modular_downscaling_model import ModelConfigurator
from losses import LossConfigurator
from training.localized_linear_model import TrainingConfigurator


def get_model_indices(geometry, step=3, offset=1):
    s = 0
    indices = []
    if isinstance(geometry, dict):
        mask = geometry[list(geometry.keys())[0]].mask
    else:
        mask = geometry.mask
    for i, row in enumerate(mask):
        c = int(np.sum(1 - row))
        if i % step == offset % step:
            current_indices = np.arange(c) + s
            indices.append(current_indices[offset::step])
        s += c
    indices = np.concatenate(indices).astype(int)
    print('[INFO] Model indices:', indices)
    return indices


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

print('[INFO] Building model.')

model = ModelConfigurator(grids=grids).build_model(config['model'])
model.set_model_index(
    datasets['training'].geometry_lr, datasets['training'].geometry_hr,
    # model_index=get_model_indices(datasets['training'].geometry_hr)
)

# plt.figure()
# plt.scatter(model.model_index_lon, model.model_index_lat)
# plt.show()

print(model)

datasets = {
    step_name: NearestNeighborData(
        datasets[step_name],
        model_index=model.model_index,
        k=model.num_nearest_neighbors_lr,
        name=step_name
    )
    for step_name in datasets.keys()
}

losses = LossConfigurator().build_losses(config['losses'])

config['training'].update({'residual_interpolation_mode': None})

training_process = TrainingConfigurator().build_training_process(
    grids,
    datasets,
    losses,
    model,
    config['training']
)

training_process.save_config(config)

training_process.run()
