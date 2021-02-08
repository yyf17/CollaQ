from smac.env import StarCraft2Env
# # from smac.env.starcraft2.maps import get_map_params
# from smac.env.starcraft2.maps import smac_maps
# def get_map_params(map_name):
#     map_param_registry = smac_maps.get_smac_map_registry()
#     # print(map_param_registry)
#     # print(map_name)
#     return map_param_registry[map_name]

import atexit

class SC2_CollaQ(StarCraft2Env):
    def __init__(self,**kwargs):
        print("enter SC2_CollaQ init")
        super(SC2_CollaQ, self).__init__(**kwargs)

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())
    def get_env_info(self):
        print("enter SC2_CollaQ get_env_info")
        env_info = super(SC2_CollaQ, self).get_env_info()
        env_info_extra = {
                "obs_alone_shape": self.get_obs_alone_size(),
             }     
        env_info.update(env_info_extra)
        print("leave SC2_CollaQ get_env_info")

        return env_info