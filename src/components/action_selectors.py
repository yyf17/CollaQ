import torch as th
from torch.distributions import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)
    
    #modify
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            if self.args.git_root in ["DOP"]:
                picked_actions = Categorical(masked_policies).sample().long()

                random_numbers = th.rand_like(agent_inputs[:, :, 0])
                pick_random = (random_numbers < self.epsilon).long()
                random_actions = Categorical(avail_actions.float()).sample().long()
                picked_actions = pick_random * random_actions + (1 - pick_random) * picked_actions
            else:
                picked_actions = Categorical(masked_policies).sample().long()
        
        if self.args.git_root in ["DOP"]:
            if not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
                return self.select_action(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    #modify
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        if self.args.git_root in ["DOP"]:
            if not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
                print((th.gather(avail_actions, dim=2, index=random_actions.unsqueeze(2)) <= 0.99).squeeze())
                print((th.gather(avail_actions, dim=2, index=masked_q_values.max(dim=2)[1].unsqueeze(2)) <= 0.99).squeeze())
                print((th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) <= 0.99).squeeze())

                print('Action Selection Error')
                # raise Exception
                return self.select_action(agent_inputs, avail_actions, t_env, test_mode) 

        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class InfluenceBasedActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.e_mode = args.e_mode

    def select_action(self, agents_inputs_alone, agents_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        if self.e_mode == "negative_sample":
            # mask actions that are excluded from selection
            masked_q_values_alone = -agents_inputs_alone.clone()
            masked_q_values_alone[avail_actions == 0.0] = -float("inf")  # should never be selected!
            # mask actions that are excluded from selection
            masked_q_values = agents_inputs.clone()
            masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        elif self.e_mode == "exclude_max":
            # mask actions that are excluded from selection
            masked_q_values_alone = agents_inputs_alone.clone()
            masked_q_values_alone[avail_actions == 0.0] = -float("inf")  # should never be selected!
            # mask actions that are excluded from selection
            masked_q_values = agents_inputs.clone()
            masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

            # Get rid off the top value
            masked_q_values_alone_max = th.argmax(masked_q_values_alone, dim=-1, keepdim=True)
            masked_q_values_alone_max_oh = th.zeros(masked_q_values_alone.shape).cuda()
            masked_q_values_alone_max_oh.scatter_(-1, masked_q_values_alone_max, 1)
            masked_q_values_alone[masked_q_values_alone_max_oh == 1] = -1
            masked_q_values_alone[masked_q_values_alone_max_oh == 0] = 0
            masked_q_values_alone = masked_q_values_alone * (th.sum(avail_actions, dim=-1, keepdim=True) != 1)
            masked_q_values_alone[masked_q_values_alone == -1] = -float("inf")
            masked_q_values_alone[avail_actions == 0.0] = -float("inf")


        random_numbers = th.rand_like(agents_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()
        #TODO: these numbers are fixed now
        if t_env > 1000000:
            random_numbers = th.rand_like(agents_inputs[:, :, 0])
            pick_alone = (random_numbers < self.args.e_prob).long()
            alone_actions = Categorical(logits=masked_q_values_alone.float()).sample().long()
            final_random_actions = pick_alone * alone_actions + (1 - pick_alone) * random_actions
        else:
            final_random_actions = random_actions

        picked_actions = pick_random * final_random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["influence"] = InfluenceBasedActionSelector

# this is for WQMIX
class PolicyEpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        # Was there so I used it
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_qs, agent_pis, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        # masked_q_values = agent_qs.clone()
        # masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_qs[:,:,0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        # max_action = th.abs(masked_q_values - agent_pis).argmin(dim=2)
        masked_agent_pis = agent_pis.clone()
        masked_agent_pis[avail_actions == 0.0] = -float("inf")
        max_action = masked_agent_pis.argmax(dim=2)
        picked_actions = pick_random * random_actions + (1 - pick_random) * max_action
        return picked_actions

REGISTRY["policy_epsilon_greedy"] = PolicyEpsilonGreedyActionSelector
#  for project LICA
# --------------------------------------------------------#
# NOTE: We added this Gumbel Action Selector  for LICA
# --------------------------------------------------------#
class GumbelSoftmax(OneHotCategorical):

    def __init__(self, logits, probs=None, temperature=1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -th.log( -th.log( U + self.eps ) )

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return th.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (th.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()


def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy()

# --------------------------------------------------------#
# NOTE: We added this Gumbel Action Selector
# --------------------------------------------------------#
class GumbelSoftmaxMultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_logits, avail_actions, t_env, test_mode=False):
        masked_policies = agent_logits.clone()

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = GumbelSoftmax(logits=masked_policies).sample()
            picked_actions = th.argmax(picked_actions, dim=-1).long()

        return picked_actions

REGISTRY["gumbel"] = GumbelSoftmaxMultinomialActionSelector

# this is for RODE
class SoftEpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              epsilon_anneal_time_exp=args.epsilon_anneal_time_exp,
                                              role_action_spaces_update_start=args.role_action_spaces_update_start,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    #modify
    def select_action(self, agent_inputs, avail_actions, role_avail_actions, t_env, test_mode=False):
    # def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, role_avail_actions=None):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        d_avail_actions = avail_actions * role_avail_actions
        masked_q_values[d_avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        ind = th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99
        if not ind.all():
            # print(">>> Action Selection Error")
            ind = ind.squeeze().long()
            picked_actions = picked_actions * ind + (1 - ind) * random_actions
        return picked_actions

# for project RODE
REGISTRY["soft_epsilon_greedy"] = SoftEpsilonGreedyActionSelector


# for project DOP
# from .action_selectors_DOP import MultinomialActionSelector as MultinomialActionSelector_DOP
# from .action_selectors_DOP import EpsilonGreedyActionSelector as EpsilonGreedyActionSelector_DOP

# REGISTRY["multinomial_DOP"] = MultinomialActionSelector_DOP

# REGISTRY["epsilon_greedy_DOP"] = EpsilonGreedyActionSelector_DOP



# for project ROMA 

# from .action_selectors_DOP import MultinomialActionSelector as MultinomialActionSelector_ROMA
# from .action_selectors_DOP import EpsilonGreedyActionSelector as EpsilonGreedyActionSelector_ROMA

# REGISTRY["multinomial_ROMA"] = MultinomialActionSelector_ROMA

# REGISTRY["epsilon_greedy_ROMA"] = EpsilonGreedyActionSelector_ROMA