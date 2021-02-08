import numpy as np
import os
import sys
from copy import deepcopy
#-------    指定可见的GPUS  ---
#       --gpus=1,2,3
def get_GPU(arg_name="--gpus"):
    #    --gpus=0,1,2,3
    cmd_params = deepcopy(sys.argv)
    print('cmd_params:',cmd_params)
    gpus_str = None
    for _i, _v in enumerate(cmd_params):
        if _v.split("=")[0] == arg_name:
            gpus_str = _v.split("=")[1]
            del cmd_params[_i]
            break
    print("gpus_str",gpus_str)
    assert gpus_str is not None ,"GPU is not specify"
    return gpus_str

def set_GPU_ids(ids_str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ids_str

set_GPU_ids(get_GPU())
#---------------------------------------------------------
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

import torch as th
from utils.logging import get_logger
import yaml

from runs import REGISTRY as run_REGISTRY
# from run import run


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework

    run_REGISTRY[config["git_root"]](_run, config, _log)
    # run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def _get_cmd_param(params, arg_name):
    # this param can add in cmd and delete it after read them
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break
    return config_name
if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # get gpus list
    gpus = _get_cmd_param(params, "--gpus")

    # get the git root name from command line
    git_root_name = _get_cmd_param(params, "--git-root")


    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), f"config/{git_root_name}", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)


    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", f"{git_root_name}/envs")
    alg_config = _get_config(params, "--config", f"{git_root_name}/algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)


    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

