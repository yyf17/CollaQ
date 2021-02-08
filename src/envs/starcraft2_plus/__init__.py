from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#修改
from smac.env import StarCraft2Env



# #---------     for maps module
# from .maps import map_param_registry,get_map_params,map_present,SMACMap

# for name in map_param_registry.keys():
#     globals()[name] = type(name, (SMACMap,), dict(filename=name))

#---------     for current dir
from .custom_scenarios import custom_scenario_registry
from .join1 import Join1Env
from .starcraft2_MAVEN import SC2 as StarCraft2Env_MAVEN
from .starcraft2_NDQ import StarCraft2Env as StarCraft2Env_NDQ
from .starcraft2_ROMA import StarCraft2Env as StarCraft2Env_ROMA
from .starcraft2_not_0 import StarCraft2Not0Env
from .starcraft2_set_1 import StarCraft2Set1Env
from .starcraft2_sort import StarCraft2SortEnv
from .starcraft2custom import StarCraft2CustomEnv
from .tracker1 import Tracker1Env



__all__ = [
    "StarCraft2Env", 
    "custom_scenario_registry",
    "Join1Env",
    "StarCraft2Env_MAVEN",
    "StarCraft2Env_NDQ",
    "StarCraft2Env_ROMA",
    "StarCraft2Not0Env",
    "StarCraft2Set1Env",
    "StarCraft2SortEnv",
    "StarCraft2CustomEnv",
    "Tracker1Env", 
]

