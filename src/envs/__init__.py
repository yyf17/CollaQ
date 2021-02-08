from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv
# from smac.env import StarCraft2Env
# from .starcraft2_plus import map_param_registry,StarCraft2Env

from .sc2_CollaQ import SC2_CollaQ
from .sc2_QPLEX import SC2_QPLEX
from .sc2_ASN import SC2_ASN


from .starcraft2_plus import StarCraft2Env
from .starcraft2_plus import StarCraft2Env_MAVEN
from .starcraft2_plus import StarCraft2Env_NDQ
from .starcraft2_plus import StarCraft2Env_ROMA
from .starcraft2_plus import StarCraft2Not0Env
from .starcraft2_plus import StarCraft2Set1Env
from .starcraft2_plus import StarCraft2SortEnv
from .starcraft2_plus import StarCraft2CustomEnv

from .starcraft2_plus import Tracker1Env
from .starcraft2_plus import Join1Env

from .starcraft2_plus import custom_scenario_registry                     #for project AIQMIX
from .starcraft2_plus import custom_scenario_registry as sc_scenarios     #for project AIQMIX
                          #for project MAVEN

from .firefighters import FireFightersEnv              #for project AIQMIX
from .firefighters import scenarios as ff_scenarios    #for project AIQMIX

from .matrix import Matrix_game1Env
from .matrix import Matrix_game2Env
from .matrix import Matrix_game3Env
from .mmdp_game_1 import mmdp_game1Env

from .matrix_game import MatrixGame              # in project WQMIX
from .matrix_game import NStepMatrixGame         #  in project MAVEN
from .stag_hunt import StagHunt                  # in project WQMIX


# from .gfootball import GoogleFootballEnv      # in project ROMA




def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

REGISTRY = {
    #     sc2
    "sc2": partial(env_fn, env=StarCraft2Env),
    "sc2_CollaQ": partial(env_fn, env=SC2_CollaQ),
    "sc2_QPLEX": partial(env_fn, env=SC2_QPLEX),
    "sc2_NDQ": partial(env_fn, env=StarCraft2Env_NDQ),
    "sc2_ROMA": partial(env_fn, env=StarCraft2Env_ROMA),
    "sc2_ASN": partial(env_fn, env=SC2_ASN),
    "sc2_MAVEN":partial(env_fn, env=StarCraft2Env_MAVEN),  #for project MAVEN
    "sc2custom":partial(env_fn, env=StarCraft2CustomEnv),  #for project AIQMIX
    "sc2_sort": partial(env_fn, env=StarCraft2SortEnv),
    "sc2_not_0": partial(env_fn, env=StarCraft2Not0Env),
    "sc2_set_1": partial(env_fn, env=StarCraft2Set1Env),
    #    matrix
    "matrix_game_1": partial(env_fn, env=Matrix_game1Env),
    "matrix_game_2": partial(env_fn, env=Matrix_game2Env),
    "matrix_game_3": partial(env_fn, env=Matrix_game3Env),
    "matrix_game":partial(env_fn, env=MatrixGame),        # for project WQMIX
    "mmdp_game_1": partial(env_fn, env=mmdp_game1Env),
    "nstep_matrix":partial(env_fn, env=NStepMatrixGame),  # for project WQMIX
    "stag_hunt":partial(env_fn, env=StagHunt),            # for project WQMIX
    "Tracker1Env":partial(env_fn, env=Tracker1Env),   #  for project NDQ
    "join1":partial(env_fn, env=Join1Env),        #  for project NDQ
    "firefighters":partial(env_fn, FireFightersEnv),       #for project AIQMIX
    # "gf":partial(env_fn, env=GoogleFootballEnv),        # in project ROMA
}




s_REGISTRY = {}
s_REGISTRY.update(ff_scenarios)  #for project AIQMIX
s_REGISTRY.update(sc_scenarios)  #for project AIQMIX





#--------------------------------------------------------------------------
# 环境变量
# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
print('set SC2PATH')
os.environ.setdefault("SC2PATH","/data/SC_data/SC2_4_6_2_69232/3rdparty/StarCraftII")
#--------------------------------------------------------------------------
# print(map_param_registry)

# if __name__ == "__main__":
#     env = REGISTRY['sc2']()
    
