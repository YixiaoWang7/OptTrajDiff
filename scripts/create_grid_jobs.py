import json
from argparse import ArgumentParser

from sklearn.model_selection import ParameterGrid


def create_grid_jobs(config_file):
    """
    Create a list of jobs given a config file

    Format of config_file:
    ```json
    {
        "executable": "python train_diffnet.py",
        "fixed_params" : {
            "root": "/home/username/data",
            "train_batch_size": 32,
            "val_batch_size": 32,
            "test_batch_size": 32,
        },
        "grid_params" : {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    }
    ```
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    fixed_params = config["fixed_params"]
    grid_params = config["grid_params"]
    executable = config["executable"]

    grid_jobs = []
    for grid_param in ParameterGrid(grid_params):
        job = {}
        params = {**fixed_params, **grid_param}

        job = executable + " " + " ".join([f"--{k} {v}" for k, v in params.items()])
        grid_jobs.append(job)

    return grid_jobs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)

    args = parser.parse_args()

    # Get directory of config file
    output_file = "grid_jobs.txt"
    grid_jobs = create_grid_jobs(args.config_file)
    with open(output_file, "w") as f:
        f.write("\n".join(grid_jobs))