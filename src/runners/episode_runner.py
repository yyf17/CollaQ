from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from utils.draw_message_distributions import draw_message_distributions

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import shutil
import copy

    # 完成合并   x   ✓
#  - ASN-------[x]  
#  - AIQMIX----[x]
#  - LICA------[x]
#  - MAVEN-----[x]
#  - WQMIX-----[x]
#  - QTRAN-----[x]
#  - QPLEX-----[✓]  
#  - CollaQ----[x]
#  - ROMA------[✓]
#  - NDQ-------[✓]
#  - RODE------[✓]
#  - DOP-------[x]

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        #  about env -----------------------------
        if self.args.git_root in ["WQMIX","AIQMIX"]:
            if 'sc2' or 'firefighters' in self.args.env:
                self.env = env_REGISTRY[self.args.env](**self.args.env_args)
            else:
                self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)
        else:    
            #    标准
            #   "ASN", "ROMA","RODE","QPLEX","NDQ"
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        #-----------------------------------------------

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        #add for RODE
        if self.args.git_root in ["RODE"]:
            self.verbose = args.verbose

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device, git_root=self.args.git_root)
        self.mac = mac
    
        # add for MAVEN
        if self.args.git_root in ["MAVEN"]:
            # Setup the noise distribution sampler
            if self.args.noise_bandit:
                if self.args.bandit_policy:
                    self.noise_distrib = enza(self.args, logger=self.logger)
                else:
                    self.noise_distrib = RBandit(self.args, logger=self.logger)
            else:
                self.noise_distrib = Uniform(self.args)

            self.noise_returns = {}
            self.noise_test_won = {}
            self.noise_train_won = {}

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()
    
    #modify
    def reset(self, *args, **kwargs):
        #*args是可变参数，args接收的是一个tuple;
        # **kwargs是关键字参数，kw接收的是一个dict
        self.batch = self.new_batch()
        self.env.reset(*args, **kwargs)
        # self.env.reset()    #  "ROMA" "RODE","QPLEX","NDQ"
        self.t = 0

    def run(self, test_mode=False, test_scenario=None, index=None, vid_writer=None, test_uniform=False, t_episode=0,  thres=0., prob=0., is_clean=False ):
        """
        test_mode: whether to use greedy action selection or sample actions
        test_scenario: whether to run on test scenarios. defaults to matching test_mode.
        vid_writer: imageio video writer object
        test_uniform : for MAVEN
        t_episode : for RODE
        thres:      for NDQ
        prob:       for NDQ
        is_clean:   for NDQ
        """

        #-------------------------------------------------
        # ["ASN","AIQMIX","LICA","MAVEN","WQMIX","QTRAN","QPLEX","CollaQ","ROMA","NDQ","RODE","DOP"]
        if self.args.git_root in ["AIQMIX"]:
            
            self.reset(test_mode=test_mode, test_scenario=test_scenario, index=index)
        elif self.args.git_root in ["MAVEN"]:
            self.reset(test_uniform)
        else:                #     ["LICA", "CollaQ", "pyMARL","ASN","ROMA","RODE","QPLEX","NDQ"]
            self.reset()

        #-------------------------------------------------


        if self.args.git_root in ["AIQMIX"]:
            if test_scenario is None:
                test_scenario = test_mode
            if vid_writer is not None:
                vid_writer.append_data(self.env.render())

        if self.args.git_root in ["NDQ"]:
            if is_clean:
                self.env.clean()
                self.mac.clean()
            if self.env.is_print_once and self.mac.is_print_once and self.args.test_is_print_once:
                return

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        if self.args.git_root in ["ROMA"]:
            if self.args.mac == "separate_mac":
                self.mac.init_latent(batch_size=self.batch_size)

        if self.args.git_root in ["ASN"]:
            legal_action_set = []
        # make sure things like dropout are disabled
        # if self.args.git_root == "AIQMIX":
        #     self.mac.eval()

        if self.args.git_root in ["ROMA"]:
            if self.args.mac == "separate_mac":
                self.mac.init_latent(batch_size=self.batch_size)

        if self.args.git_root in ["RODE"]:
            replay_data = []
            if self.verbose:
                if t_episode < 2:
                    save_path = os.path.join(self.args.local_results_path,
                                            "pic_replays",
                                            self.args.unique_token,
                                            str(t_episode))
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                    os.makedirs(save_path)
                    role_color = np.array(['r', 'y', 'b', 'c', 'm', 'g'])
                    print(self.mac.role_action_spaces.detach().cpu().numpy())
                    logging.getLogger('matplotlib.font_manager').disabled = True
                all_roles = []

        if self.args.git_root in ["NDQ"]:
            if self.args.draw_message_distributions:
                if self.args.env == 'tracker1':
                    tt_inputs = th.Tensor([[-1., 1, 1, 0, 0],
                                        [-1., 1, 1, 0, 0],
                                        [1., 0, 0, 1, 0],
                                        [1., 1, 0, 1, 0],
                                        [0., -1, 0, 0, 1],
                                        [1., -1, 0, 0, 1]])
                    (mu, sigma), messages, _ = self.mac._communicate(2, tt_inputs)
                    draw_message_distributions(self.args, mu, sigma)
                    return


        while not terminated:

            pre_transition_data = self._get_pre_transition_data()

            if self.args.git_root in ["RODE"]:
                if self.verbose:
                    # These outputs are designed for SMAC
                    ally_info, enemy_info = self.env.get_structured_state()
                    replay_data.append([ally_info, enemy_info])                

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.args.git_root in ["LICA","ASN","CollaQ","ROMA","QPLEX"]:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                cpu_actions = actions.to("cpu").numpy()

            elif self.args.git_root in ["RODE"]:
                actions, roles, role_avail_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                cpu_actions = actions.to("cpu").numpy()
                cpu_roles = roles.to("cpu").numpy()
                cpu_role_avail_actions = role_avail_actions.to("cpu").numpy()
                self.batch.update({"role_avail_actions": cpu_role_avail_actions.tolist()}, ts=self.t)               

            elif self.args.git_root in ["NDQ"]:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                                thres=thres, prob=prob)
                cpu_actions = actions.to("cpu").numpy()              

            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, env=self.env)
                cpu_actions = actions.to("cpu").numpy()
            

            if self.args.git_root in ["RODE"]:
                if self.verbose:
                    roles_detach = roles.detach().cpu().squeeze().numpy()
                    ally_info = replay_data[-1][0]
                    p_roles = np.where(ally_info['health'] > 0, roles_detach,
                                    np.array([-5 for _ in range(self.args.n_agents)]))

                    all_roles.append(copy.deepcopy(p_roles))

                    if t_episode < 2:
                        figure = plt.figure()

                        print(self.t, p_roles)
                        ally_health = ally_info['health']
                        ally_health_max = ally_info['health_max']
                        if 'shield' in ally_info.keys():
                            ally_health += ally_info['shield']
                            ally_health_max += ally_info['shield_max']
                        ally_health_status = ally_health / ally_health_max
                        plt.scatter(ally_info['x'], ally_info['y'], s=20*ally_health_status, c=role_color[roles_detach])
                        for agent_i in range(self.args.n_agents):
                            plt.text(ally_info['x'][agent_i], ally_info['y'][agent_i], '{:d}'.format(agent_i+1), c='y')

                        enemy_info = replay_data[-1][1]
                        enemy_health = enemy_info['health']
                        enemy_health_max = enemy_info['health_max']
                        if 'shield' in enemy_info.keys():
                            enemy_health += enemy_info['shield']
                            enemy_health_max += enemy_info['shield_max']
                        enemy_health_status = enemy_health / enemy_health_max
                        plt.scatter(enemy_info['x'], enemy_info['y'], s=20*enemy_health_status, c='k')
                        for enemy_i in range(len(enemy_info['x'])):
                            plt.text(enemy_info['x'][enemy_i], enemy_info['y'][enemy_i], '{:d}'.format(enemy_i+1))

                        plt.xlim(0, 32)
                        plt.ylim(0, 32)
                        plt.title('t={:d}'.format(self.t))
                        pic_name = os.path.join(save_path, str(self.t) + '.png')
                        plt.savefig(pic_name)
                        plt.close()

            if self.args.git_root in ["ASN"]:
                for idx, use_action in enumerate(cpu_actions[0]):
                    legal_action_set.append(self.batch["avail_actions"][:, self.t][0][idx][use_action])


            # reward, terminated, env_info = self.env.step(actions[0])
            # reward, terminated, env_info = self.env.step(actions[0].cpu())
            reward, terminated, env_info = self.env.step(cpu_actions[0])  #Fix memory leak #73

            if vid_writer is not None:
                vid_writer.append_data(self.env.render())

            episode_return += reward

            if self.args.git_root in ["RODE"]:
                post_transition_data = {
                    "actions": cpu_actions,
                    "roles": cpu_roles,
                    "role_avail_actions": cpu_role_avail_actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }

            else:
                post_transition_data = {
                    "actions": cpu_actions, #Fix memory leak #73
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        if self.args.git_root in ["ASN"] and test_mode:
            print("legal_action_set:", np.sum(legal_action_set), len(legal_action_set))

        last_data = self._get_pre_transition_data()

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.args.git_root in ["LICA","ASN","CollaQ","ROMA","QPLEX"]:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            self.batch.update({"actions": cpu_actions}, ts=self.t)

        elif self.args.git_root in ["NDQ"]:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                            thres=thres, prob=prob)
            cpu_actions = actions.to("cpu").numpy()
            self.batch.update({"actions": cpu_actions}, ts=self.t)         

        elif self.args.git_root in ["RODE"]:
            actions, roles, role_avail_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            cpu_roles = roles.to("cpu").numpy()
            cpu_role_avail_actions = role_avail_actions.to("cpu").numpy()
            self.batch.update({"actions": cpu_actions, "roles": cpu_roles, "role_avail_actions": cpu_role_avail_actions}, ts=self.t)

        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, env=self.env)
            cpu_actions = actions.to("cpu").numpy()
            self.batch.update({"actions": cpu_actions}, ts=self.t)


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if self.args.git_root in ["RODE"]:
            if self.verbose:
                return self.batch, np.array(all_roles)

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
    
    # # modify
    # def func_reset(self,obj,**kwargs):
    #     # ["ASN","AIQMIX","LICA","MAVEN","WQMIX","QTRAN","QPLEX","CollaQ","ROMA","NDQ","RODE","DOP"]
    #     test_mode = kwargs["test_mode"]
    #     test_scenario = kwargs["test_scenario"]
    #     index = kwargs["index"]
    #     #-------------------------------------------------
    #     if self.args.git_root in ["AIQMIX"]:
    #         obj.reset(test_mode=test_mode, test_scenario=test_scenario, index=index)
    #     elif self.args.git_root in [""]:
    #         obj.reset(test_mode=test_mode)
    #     else:                #     ["LICA", "CollaQ", "pyMARL","ASN"]
    #         obj.reset()
    #     #-------------------------------------------------
    # modify
    def _get_pre_transition_data(self):
        # if self.args.get("entity_scheme",False):
        if getattr(self.args,"entity_scheme",False):
            obs_mask, entity_mask = self.env.get_masks()
            pre_transition_data = {
                "entities": [self.env.get_entities()],
                "obs_mask": [obs_mask],
                "entity_mask": [entity_mask],
                "avail_actions": [self.env.get_avail_actions()]
            }
        else:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            #------------------------------------------------------
            if self.args.git_root == "CollaQ":
                #for CollaQ  only
                xx = {
                    "obs_alone": [self.env.get_obs_alone()],
                    }  
                # 通过update的方法添加
                pre_transition_data.update(xx)          
            #------------------------------------------------------
        return pre_transition_data
    
    # modify for ASN
    def get_move_enemy_agent_size(self):
        return self.env.get_move_feats_size(), self.env.get_enemy_feats_size(), self.env.get_agent_feats_size()
    # add for MAVEN
    def _update_noise_returns(self, returns, noise, stats, test_mode):
        for n, r in zip(noise, returns):
            n = int(np.argmax(n))
            if n in self.noise_returns:
                self.noise_returns[n].append(r)
            else:
                self.noise_returns[n] = [r]
        if test_mode:
            noise_won = self.noise_test_won
        else:
            noise_won = self.noise_train_won

        if stats != [] and "battle_won" in stats[0]:
            for n, info in zip(noise, stats):
                if "battle_won" not in info:
                    continue
                bw = info["battle_won"]
                n = int(np.argmax(n))
                if n in noise_won:
                    noise_won[n].append(bw)
                else:
                    noise_won[n] = [bw]

    # add for MAVEN
    def _log_noise_returns(self, test_mode, test_uniform):
        if test_mode:
            max_noise_return = -100000
            for n,rs in self.noise_returns.items():
                n_item = n
                r_mean = float(np.mean(rs))
                max_noise_return = max(r_mean, max_noise_return)
                self.logger.log_stat("{}_noise_test_ret_u_{:1}".format(n_item, test_uniform), r_mean, self.t_env)
            self.logger.log_stat("max_noise_test_ret_u_{:1}".format(test_uniform), max_noise_return, self.t_env)

        noise_won = self.noise_test_won
        prefix = "test"
        if test_uniform:
            prefix += "_uni"
        if not test_mode:
            noise_won = self.noise_train_won
            prefix = "train"
        if len(noise_won.keys()) > 0:
            max_test_won = 0
            for n, rs in noise_won.items():
                n_item = n #int(np.argmax(n))
                r_mean = float(np.mean(rs))
                max_test_won = max(r_mean, max_test_won)
                self.logger.log_stat("{}_noise_{}_won".format(n_item, prefix), r_mean, self.t_env)
            self.logger.log_stat("max_noise_{}_won".format(prefix), max_test_won, self.t_env)
        self.noise_returns = {}
        self.noise_test_won = {}
        self.noise_train_won = {}

    # add for MAVEN
    def save_models(self, path):
        if self.args.noise_bandit:
            self.noise_distrib.save_model(path)