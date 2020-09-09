import os
import json

from utils import NumpyEncoder

data_specs = {
    "directory": "/data/dir",  # enter data storage directory here
    "lr_filter_name": "input",
    "hr_filter_name": "target",
    "region": "alps",
    "training": {
        "time_range": [[2016, 3], [2019, 9]],
        "patching": True,
        "patch_size": [24, 36]
    },
    "validation": {
        "time_range": [[2016, 6], [2017, 5]],
        "patching": False,
        "patch_size": [24, 36]
    }
}

loss_specs = {
    "use_mask": True,
    "batch_reduction": "sum",
    "training": [
        {
            "use_scalings": False,
            "losses": ["MSE", "MAE", "L1", "Cos-D"]
        },
        {
            "use_scalings": True,
            "losses": ["MSE", "MAE", "L1", "Cos-D"]
        }
    ],
    "validation": [
        {
            "use_scalings": False,
            "losses": ["MSE", "MAE", "L1", "Cos-D"]
        },
        {
            "use_scalings": True,
            "losses": ["MSE", "MAE", "L1", "Cos-D"]
        }
    ]
}

training_specs = {
    "directory": "results/dir",  # enter output storage directory here
    "initialization_mode": "orthogonal",
    "gpu": 0,
    "epochs": 80,
    "batch_size": 400,
    "saving_period": 10,
    "optimizer": {
        "type": "adam",
        "learning_rate": .1,
        "betas": [
            0.9,
            0.999
        ],
        "weight_decay": 0.0001
    },
    "scheduler": {
        "type": "plateau",
        "gamma": 0.1,
        "steps": 5
    },
    "debugging": {
        "print_network_details": True,
        "print_config": True,
        "print_directories": True
    }
}

json_dir = 'json/dir' # enter folder path for json configs here
output_dir = 'open'

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

config_file_list = [f for f in sorted(os.listdir(json_dir)) if f.endswith('.json')]

for current_file in config_file_list:
    print('[INFO] Processing file {}'.format(current_file))
    with open(os.path.join(json_dir, current_file), encoding='utf-8') as f:
        current_config = json.load(f)
    current_config["losses"].update(loss_specs)
    current_config["training"].update(training_specs)
    file_name = current_file.split('.')[0]
    current_config["model"].update({"name": file_name})
    for year in [2016, 2017, 2018]:
        data_specs['validation'].update({'time_range': [[year, 6], [year + 1, 5]]})
        current_config["data"].update(data_specs)
        current_file_name = '{}_{}.json'.format(file_name, year)
        with open(os.path.join(output_dir, current_file_name), 'w') as f:
            json.dump(current_config, f, indent=4, cls=NumpyEncoder)
