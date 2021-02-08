# pipeline of one run
# (1) runner = r_REGISTRYargs.runner(args=args, logger=logger)
# (2) mac = mac_REGISTRYargs.mac(buffer.scheme, groups, args)
# (3) bind runner and mac runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
# (4) learner = le_REGISTRYargs.learner(mac, buffer.scheme, logger, args)
import yaml
import os
from copy import deepcopy
import collections
import yaml
import os
from types import SimpleNamespace as SN
import torch as th

from utils.logging import get_logger
from components.transforms import OneHot
from components.episode_buffer import ReplayBuffer

from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY

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
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
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

def get_algorithms_graph(alg_config='qtran',env_config='sc2',map_name='MMM2,'):
    params = [f'--config={alg_config}',f'--env-config={env_config}',f'with env_args.map_name={map_name}']
    #print(params)
    # (1) load default config
    with open(os.path.join(os.path.dirname(__file__), "config", 'default.yaml'), "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    # (2) Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    #print(config_dict)
    # (3) config_dict 传给 args
    args = SN(**config_dict)
    args.device = "cuda" if args.use_cuda else "cpu"
    # (4) 创建logger
    logger = get_logger()

    # (5) 创建runner
    runner = r_REGISTRY[args.runner](args=args,logger=logger)
    # (6) 创建env_info 
    env_info = runner.get_env_info()

    # (7) 默认的scheme
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        # This is added for Q_alone
        "obs_alone": {"vshape": env_info["obs_alone_shape"], "group": "agents"},
    }
    # (8) 修改args
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    # (9) 创建预处理函数
    preprocess = {
    "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    # (10) 创建groups
    groups = {
    "agents": args.n_agents
    }
    # (11) 创建buffer
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device)
    # (12) 修改args
    args.obs_last_action = False
    # (13) 创建mac
    mac = mac_REGISTRY[args.mac](buffer.scheme,groups,args)
    # (14) 绑定runnner和mac
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    # (15) 创建learner
    learner = le_REGISTRY[args.learner](mac,buffer.scheme,logger,args)
    # print('learner.mac.agent')
    # print(learner.mac.agent)
    # print('learner.mixer')
    # print(learner.mixer)
    return learner



# if __name__ == '__main__':
#     get_algorithms_graph()


