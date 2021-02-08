from smac.env import StarCraft2Env
import atexit

class SC2_ASN(StarCraft2Env):
    def __init__(self,**kwargs):
        print("enter SC2_ASN init")
        super(SC2_ASN, self).__init__(**kwargs)

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())
    
    def get_enemy_feats_size(self):
        nf_en = 4 + self.unit_type_bits

        nf_en += 1 if getattr(self, "obs_enemies_attacked_num", False) else 0

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return nf_en

    def get_agent_feats_size(self):
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        return nf_al

    def get_move_feats_size(self):
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_own_feats_szie(self):
        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally

        return own_feats