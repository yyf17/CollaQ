import numpy as np


class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 #modify for RODE
                 time_length_exp=None,
                 role_action_spaces_update_start=None,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        #modify
        self.role_action_spaces_update_start = role_action_spaces_update_start
        if self.role_action_spaces_update_start is not None:
            self.reset = True
            self.time_length_exp = time_length_exp
            self.start_t = 0


        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        #modify
        if self.role_action_spaces_update_start is not None:
            if T > self.role_action_spaces_update_start and self.reset:
                self.reset = False
                self.time_length = self.time_length_exp
                self.delta = (self.start - self.finish) / self.time_length
                self.start_t = T

        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass
