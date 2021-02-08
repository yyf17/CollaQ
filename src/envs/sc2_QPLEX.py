from smac.env import StarCraft2Env

import atexit

class SC2_QPLEX(StarCraft2Env):
    def __init__(self,**kwargs):
        # print('Enter SC2_QPLEX:')
        # print(kwargs)
        # {'continuing_episode': False, 'difficulty': '7', 'game_version': None, 'map_name': ['3s5z'], 'move_amount': 2, 'obs_all_health': True, 'obs_instead_of_state': False, 'obs_last_action': False, 'obs_own_health': True, 'obs_pathing_grid': False, 'obs_terrain_height': False, 'obs_timestep_number': False, 'reward_death_value': 10, 'reward_defeat': 0, 'reward_negative_scale': 0.5, 'reward_only_positive': True, 'reward_scale': True, 'reward_scale_rate': 20, 'reward_sparse': False, 'reward_win': 200, 'replay_dir': '', 'replay_prefix': '', 'state_last_action': True, 'state_timestep_number': False, 'step_mul': 8, 'seed': 64672901, 'heuristic_ai': False, 'heuristic_rest': False, 'debug': False}
        super(SC2_QPLEX, self).__init__(**kwargs)
        
        # Qatten
        self.unit_dim = 4 + self.shield_bits_ally + self.unit_type_bits
        
        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())

    def get_env_info(self):
        # self.__init__()
        env_info = super().get_env_info()
        env_info_extra = {
             "unit_dim": self.unit_dim
             }     
        env_info.update(env_info_extra)

        return env_info