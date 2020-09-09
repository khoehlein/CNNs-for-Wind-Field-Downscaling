import json

from data import GridConfigurator, DataConfigurator
from networks.modular_downscaling_model import ModelConfigurator
from losses import LossConfigurator
from training.modular_downscaling_model import TrainingConfigurator

with open("training\\modular_downscaling_model\\scripts\\base_config.json", encoding='utf-8') as f:
    config = json.load(f)

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

training_process.run()

