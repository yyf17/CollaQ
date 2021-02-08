import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY

from components.transforms import OneHot

from numpy.random import RandomState
from math import ceil
import imageio
from os.path import basename, join, splitext
import shutil
import numpy as np
import copy as cp
import random

# from components.episode_buffer import ReplayBuffer
from components import REGISTRY as buffer_REGISTRY

#modify for AIQMIX
from envs import s_REGISTRY


def run(_run, _config, _log, pymongo_client=None):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    #------------------------------------------------------------------------------------------------------
    # if args.git_root in ["NDQ"]:
        # configure tensorboard logger
    game_name = ""
    if 'map_name' in args.env_args:
        unique_token = "{}__{}__{}".format(
            args.name,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env_args['map_name']
        )
        #AttributeError: 'list' object has no attribute 'replace'
        game_name =  args.env_args['map_name']
    else:
        unique_token = "{}__{}__{}".format(
            args.name,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env
        )
        game_name = args.env
    # else:      #   "ASN","MAVEN"
    #     # configure tensorboard logger
    #     unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    #------------------------------------------------------------------------------------------------------
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(args.local_results_path, f"tb_logs/{game_name}")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    #modify
    if args.git_root in ["WQMIX","AIQMIX"]:
        if pymongo_client is not None:
            print("Attempting to close mongodb client")
            pymongo_client.close()
            print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

#modify for AIQMIX
def evaluate_sequential(args, runner, logger=None):
    logger = logger
    #---------------------------------------------------------------------
    #modify
    if args.git_root in ["RODE"]:
        all_roles = np.array([0 for _ in range(runner.mac.n_roles)])

        for episode_i in range(args.test_nepisode):
            if args.verbose:
                _, all_role = runner.run(test_mode=True, t_episode=episode_i)
                for role_i in range(runner.mac.n_roles):
                    all_roles[role_i] += (all_role == role_i).sum()

            else:
                runner.run(test_mode=True, t_episode=episode_i)
    elif args.git_root in ["AIQMIX"]:
        #------------------------------------------------------
        vw = None
        if args.video_path is not None:
            os.makedirs(dirname(args.video_path), exist_ok=True)
            vid_basename_split = splitext(basename(args.video_path))
            if vid_basename_split[1] == '.mp4':
                vid_basename = ''.join(vid_basename_split)
            else:
                vid_basename = ''.join(vid_basename_split) + '.mp4'
            vid_filename = join(dirname(args.video_path), vid_basename)
            vw = imageio.get_writer(vid_filename, format='FFMPEG', mode='I',
                                    fps=args.fps, codec='h264', quality=10)

        if args.eval_path is not None:
            os.makedirs(dirname(args.eval_path), exist_ok=True)
            eval_basename_split = splitext(basename(args.eval_path))
            if eval_basename_split[1] == '.json':
                eval_basename = ''.join(eval_basename_split)
            else:
                eval_basename = ''.join(eval_basename_split) + '.json'
            eval_filename = join(dirname(args.eval_path), eval_basename)

        res_dict = {}

        if args.eval_all_scen:
            if 'sc2' in args.env:
                dict_key = 'scenarios'
            elif 'firefighters' in args.env:
                if args.eval_train_scen:
                    dict_key = 'train_scenarios'
                else:
                    dict_key = 'test_scenarios'
            n_scen = len(args.env_args['scenario_dict'][dict_key])
        else:
            n_scen = 1
        n_test_batches = max(1, args.test_nepisode // runner.batch_size)

        for i in range(n_scen):
            run_args = {'test_mode': True, 'vid_writer': vw,
                        'test_scenario': (not args.eval_train_scen)}
            if args.eval_all_scen:
                run_args['index'] = i
            for _ in range(n_test_batches):
                runner.run(**run_args)
            curr_stats = dict((k, v[-1][1]) for k, v in logger.stats.items())
            if args.eval_all_scen:
                curr_scen = args.env_args['scenario_dict'][dict_key][i]
                scen_str = "".join(curr_scen[0])  # assumes that unique set of agents is a unique scenario
                res_dict[scen_str] = curr_stats
            else:
                res_dict.update(curr_stats)

        if vw is not None:
            vw.close()

        if args.eval_path is not None:
            with open(eval_filename, 'w') as f:
                json.dump(res_dict, f)
        #------------------------------------------------------
    elif args.git_root in ["NDQ"]:
        if args.test_is_cut:
            if args.test_is_cut_prob:
                print('mu thres:', 0.)
                for _ in range(args.test_nepisode):
                    runner.run(test_mode=True, thres=0., is_clean=(_ == 0))
                thres = args.test_cut_prob_thres
                for prob in args.test_cut_prob_list:
                    print('mu+prob thres:', thres, prob)
                    for _ in range(args.test_nepisode):
                        runner.run(test_mode=True, thres=thres, prob=prob, is_clean=(_ == 0))
            else:
                for thres in args.test_cut_list:
                    print('mu thres:', thres)
                    for _ in range(args.test_nepisode):
                        runner.run(test_mode=True, thres=thres, is_clean=(_ == 0))
        else:
            for _ in range(args.test_nepisode):
                runner.run(test_mode=True)      
    else:   #  "LICA","MAVEN","QPLEX","DOP"
        for _ in range(args.test_nepisode):
            runner.run(test_mode=True)
    #---------------------------------------------------------------------


    if args.save_replay:
        runner.save_replay()
    
    #---------------------------------------------------------------------
    #modify for RODE
    if args.git_root in ["RODE"]:
        if args.verbose:
            role_frequency = all_roles / all_roles.sum()
            print(role_frequency)
            fre_file = os.path.join(args.local_results_path,"pic_replays",args.unique_token,"role_frequency.pkl")
            with open(fre_file, "wb") as f:
                pickle.dump(role_frequency, f)
    #---------------------------------------------------------------------

    runner.close_env()

    if logger is not None:
        logger.print_stats_summary()

def save_one_buffer(args, save_buffer, env_name, from_start=False):
    x_env_name = env_name
    if from_start:
        x_env_name += '_from_start/'
    
    path_name = args.local_results_path+'/buffer/' + x_env_name + '/buffer_' + str(args.save_buffer_id) + '/'
    if os.path.exists(path_name):
        random_name = args.local_results_path+'/buffer/' + x_env_name + '/buffer_' + str(random.randint(10, 1000)) + '/'
        os.rename(path_name, random_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    save_buffer.save(path_name)

def run_sequential(args, logger):
    #modify args for special use in AIQMIX

    if 'entity_scheme' in args.env_args:
        args.entity_scheme = args.env_args['entity_scheme']
    else:
        args.entity_scheme = False

    if args.git_root in ["AIQMIX"]:    
        if 'sc2' or 'firefighters' in args.env:
            # ensure same train/test split across seeds
            rs = RandomState(0)
            args.env_args['scenario_dict'] = s_REGISTRY[args.scenario](rs=rs)


    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    #modify for ASN
    if args.git_root in ["ASN"]:
        # modify
        _log = logger.console_logger
        move_feats_size, enemy_feats_size, agent_feats_size = runner.get_move_enemy_agent_size()
        args.move_feats_size = move_feats_size
        args.enemy_feats_size = enemy_feats_size
        args.agent_feats_size = agent_feats_size

        _log.info('\n\n ' + 'move feat size: ' + str(args.move_feats_size) + \
                '\n ememy feat size: ' + str(args.enemy_feats_size) + \
                '\n agent feat size: ' + str(args.agent_feats_size))

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]

    if args.entity_scheme:  #True
        args.entity_shape = env_info["entity_shape"]
        args.n_entities = env_info["n_entities"]
    else:
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]

    if args.git_root in ["QPLEX"]:
        args.unit_dim = env_info["unit_dim"]

        env_name = args.env
        if env_name == 'sc2':
            env_name += '/' + args.env_args['map_name'][0]







    scheme,groups,preprocess = get_scheme_groups_preprocess(args,env_info)

    if args.git_root in ["QPLEX"]:
        buffer = buffer_REGISTRY[args.component](scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                            burn_in_period=args.burn_in_period,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device, git_root=args.git_root)

        if args.is_save_buffer:
            save_buffer = buffer_REGISTRY[args.component](scheme, groups, args.save_buffer_size, env_info["episode_limit"] + 1,
                                    burn_in_period=args.burn_in_period,
                                    preprocess=preprocess,
                                    device="cpu" if args.buffer_cpu_only else args.device, git_root=args.git_root)

        if args.is_batch_rl:
            assert (args.is_save_buffer == False)
            x_env_name = env_name
            if args.is_from_start:
                x_env_name += '_from_start/'
            path_name = args.local_results_path+'/buffer/' + x_env_name + '/buffer_' + str(args.load_buffer_id) + '/'
            assert (os.path.exists(path_name) == True)
            buffer.load(path_name)
    elif args.git_root in ["DOP"]:
        buffer = buffer_REGISTRY[args.component](scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                    preprocess=preprocess,
                    device="cpu" if args.buffer_cpu_only else args.device, git_root=args.git_root)
        off_buffer = buffer_REGISTRY[args.component](scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device, git_root=args.git_root)
    else:
        buffer = buffer_REGISTRY[args.component](scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device, git_root=args.git_root)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
        # runner.cuda()   #AttributeError: 'EpisodeRunner' object has no attribute 'cuda' 

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
        if args.git_root in ["NDQ"]:
            # only for NDQ
            #---------------------------------------------------------
            timesteps.sort()

            if args.make_message_distribution_video and args.draw_message_distributions:
                save_dir = args.checkpoint_path.replace('models', 'plots')
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.mkdir(save_dir)

                for timestep in timesteps:
                    model_path = os.path.join(args.checkpoint_path, str(timestep))

                    logger.console_logger.info("Loading model from {}".format(model_path))
                    learner.load_models(model_path)
                    args.loaded_model_ts = timestep

                    episode_batch = runner.run(test_mode=False)
            else:
                if args.load_step == 0:
                    # choose the max timestep
                    timestep_to_load = max(timesteps)
                else:
                    # choose the timestep closest to load_step
                    timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

                model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

                logger.console_logger.info("Loading model from {}".format(model_path))
                learner.load_models(model_path)
                runner.t_env = timestep_to_load

                if args.evaluate or args.save_replay:
                    evaluate_sequential(args, runner)
                    return        
            #---------------------------------------------------------
        else:
            #---------------------------------------------------------
            if args.load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

            model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

            logger.console_logger.info("Loading model from {}".format(model_path))

            if args.git_root in ["AIQMIX"]:
                learner.load_models(model_path, evaluate=args.evaluate)
            else:
                learner.load_models(model_path)

            runner.t_env = timestep_to_load

            if args.evaluate or args.save_replay:

                if args.git_root in ["AIQMIX"]:
                    runner.reset(test=True) 
                else:   
                    runner.reset(test_mode=True)
                # runner.env.reset(test_mode=True)
                # Set up schemes and groups here
                env_info = runner.get_env_info()

                scheme,groups,preprocess = get_scheme_groups_preprocess(args,env_info)

                # Give runner the scheme
                runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
                mac.n_agents = env_info["n_agents"]
                
                if args.git_root in ["ASN"]:
                    pass       # AttributeError: 'AsnRNNAgent' object has no attribute 'update_n_agents'
                else:
                    mac.agent.update_n_agents(env_info["n_agents"])

                if args.git_root in ["AIQMIX"]:
                    evaluate_sequential(args, runner, logger)
                else:
                    evaluate_sequential(args, runner)

                return

                #----------------------------------------------------------------------

            #---------------------------------------------------------
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    if args.git_root in ["QPLEX"]:
        if args.env == 'matrix_game_1' or args.env == 'matrix_game_2' or args.env == 'matrix_game_3' \
            or args.env == 'mmdp_game_1':
            last_demo_T = -args.demo_interval - 1

    while runner.t_env <= args.t_max:
        if args.git_root in ["QPLEX"]:
            # only for QPLEX
            if not args.is_batch_rl:
                # Run for a whole episode at a time
                episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)

                if args.is_save_buffer:
                    save_buffer.insert_episode_batch(episode_batch)
                    if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                        save_buffer.is_from_start = False
                        save_one_buffer(args, save_buffer, env_name, from_start=True)
                    if save_buffer.buffer_index % args.save_buffer_interval == 0:
                        print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

            for _ in range(args.num_circle):
                if buffer.can_sample(args.batch_size):
                    episode_sample = buffer.sample(args.batch_size)

                    if args.is_batch_rl:
                        runner.t_env += int(th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

                    if args.env == 'mmdp_game_1' and args.learner == "q_learner_exp":
                        for i in range(int(learner.target_gap) - 1):
                            episode_sample = buffer.sample(args.batch_size)

                            # Truncate batch to only filled timesteps
                            max_ep_t = episode_sample.max_t_filled()
                            episode_sample = episode_sample[:, :max_ep_t]

                            if episode_sample.device != args.device:
                                episode_sample.to(args.device)

                            learner.train(episode_sample, runner.t_env, episode)       
            
                #--------------------------------------

        elif args.git_root in ["DOP"]:
            # only for DOP
            # critic running log
            running_log = {
                "critic_loss": [],
                "critic_grad_norm": [],
                "td_error_abs": [],
                "target_mean": [],
                "q_taken_mean": [],
                "q_max_mean": [],
                "q_min_mean": [],
                "q_max_var": [],
                "q_min_var": []
            }

            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)
            off_buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size) and off_buffer.can_sample(args.off_batch_size):
                #train critic normall
                uni_episode_sample = buffer.uni_sample(args.batch_size)
                off_episode_sample = off_buffer.uni_sample(args.off_batch_size)
                max_ep_t = max(uni_episode_sample.max_t_filled(), off_episode_sample.max_t_filled())
                uni_episode_sample = process_batch(uni_episode_sample[:, :max_ep_t], args)
                off_episode_sample = process_batch(off_episode_sample[:, :max_ep_t], args)
                learner.train_critic(uni_episode_sample, best_batch=off_episode_sample, log=running_log)

                #train actor
                episode_sample = buffer.sample_latest(args.batch_size)
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = process_batch(episode_sample[:, :max_ep_t], args)
                learner.train(episode_sample, runner.t_env, running_log)   
            #-------------------------
                   
        else:
            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)

            if args.git_root in ["NDQ"]:
                if args.draw_message_distributions:
                    return

            buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size):
                # if "training_iters" not in args:   #  TypeError: argument of type 'types.SimpleNamespace' is not iterable
                #     args["training_iters"] = 1
                # for _ in range(args.training_iters):
                for _ in range(getattr(args,"training_iters",1)):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    if args.git_root in ["LICA"]:
                        learner.train_critic_td(episode_sample, runner.t_env, episode)

                    learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env

            # runner.env.reset(test_mode=True)
            if args.git_root in ["AIQMIX"]:
                runner.reset(test=True) 
            else:   
                runner.reset(test_mode=True)
            # Set up schemes and groups here
            env_info = runner.get_env_info()

            scheme,groups,preprocess = get_scheme_groups_preprocess(args,env_info)

            # Give runner the scheme
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
            mac.n_agents = env_info["n_agents"]

            if args.git_root in ["ASN"]:
                pass       # AttributeError: 'AsnRNNAgent' object has no attribute 'update_n_agents'
            else:
                mac.agent.update_n_agents(env_info["n_agents"])

            if args.git_root in ["NDQ"]:
                #for NDQ
                for i in [90., 95., 98.]:
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, thres=i)
            else:
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

            if args.git_root in ["AIQMIX"]:
                runner.reset(test=True) 
            else:   
                runner.reset(test_mode=True)
            # Set up schemes and groups here
            env_info = runner.get_env_info()

            scheme,groups,preprocess = get_scheme_groups_preprocess(args,env_info)

            # Give runner the scheme
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
            mac.n_agents = env_info["n_agents"]

            if args.git_root in ["ASN"]:
                pass       # AttributeError: 'AsnRNNAgent' object has no attribute 'update_n_agents'
            else:
                mac.agent.update_n_agents(env_info["n_agents"])

            if args.git_root in ["MAVEN"]:
                if args.noise_bandit:
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, test_uniform=True)  

                    
                    runner.reset(test_mode=False)
                    # Set up schemes and groups here
                    env_info = runner.get_env_info()

                    scheme,groups,preprocess = get_scheme_groups_preprocess(args,env_info)

                    # Give runner the scheme
                    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
                    mac.n_agents = env_info["n_agents"]
                    
                    if args.git_root in ["ASN"]:
                        pass       # AttributeError: 'AsnRNNAgent' object has no attribute 'update_n_agents'
                    else:
                        mac.agent.update_n_agents(env_info["n_agents"])        

        # add two process for QPLEX
        # (1) mmdp_game_1
        #-------------------------------------------------------------
        if args.git_root in ["QPLEX"]:
            if args.env == 'mmdp_game_1' and \
                    (runner.t_env - last_demo_T) / args.demo_interval >= 1.0 and buffer.can_sample(args.batch_size):
                ### demo
                episode_sample = cp.deepcopy(buffer.sample(1))
                for i in range(args.n_actions):
                    for j in range(args.n_actions):
                        new_actions = th.Tensor([i, j]).unsqueeze(0).repeat(args.episode_limit + 1, 1)
                        if i == 0 and j == 0:
                            rew = th.Tensor([1, ])
                        else:
                            rew = th.Tensor([0, ])
                        if i == 1 and j == 1:
                            new_obs = th.Tensor([1, 0]).unsqueeze(0).unsqueeze(0).repeat(args.episode_limit, args.n_agents, 1)
                        else:
                            new_obs = th.Tensor([0, 1]).unsqueeze(0).unsqueeze(0).repeat(args.episode_limit, args.n_agents, 1)
                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]
                        episode_sample['actions'][0, :, :, 0] = new_actions
                        episode_sample['obs'][0, 1:, :, :] = new_obs
                        episode_sample['reward'][0, 0, 0] = rew
                        new_actions_onehot = th.zeros(episode_sample['actions'].squeeze(3).shape + (args.n_actions,))
                        new_actions_onehot = new_actions_onehot.scatter_(3, episode_sample['actions'].cpu(), 1)
                        episode_sample['actions_onehot'][:] = new_actions_onehot

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)

                        #print("action pair: %d, %d" % (i, j))
                        learner.train(episode_sample, runner.t_env, episode, show_demo=True, save_data=(i, j))
                last_demo_T = runner.t_env
                #time.sleep(1)
        #-------------------------------------------------------------
        # (2) 'matrix_game_1','matrix_game_2','matrix_game_3'
        if args.git_root in ["QPLEX"]:
            if (args.env == 'matrix_game_1' or args.env == 'matrix_game_2' or args.env == 'matrix_game_3') and \
                (runner.t_env - last_demo_T) / args.demo_interval >= 1.0 and buffer.can_sample(args.batch_size):
                ### demo
                episode_sample = cp.deepcopy(buffer.sample(1))
                for i in range(args.n_actions):
                    for j in range(args.n_actions):
                        new_actions = th.Tensor([i, j]).unsqueeze(0).repeat(args.episode_limit + 1, 1)
                        if i == 0 and j == 0:
                            rew = th.Tensor([1, ])
                        else:
                            rew = th.Tensor([0, ])
                        if i == 1 and j == 1:
                            new_obs = th.Tensor([1, 0]).unsqueeze(0).unsqueeze(0).repeat(args.episode_limit, args.n_agents, 1)
                        else:
                            new_obs = th.Tensor([0, 1]).unsqueeze(0).unsqueeze(0).repeat(args.episode_limit, args.n_agents, 1)
                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]
                        episode_sample['actions'][0, :, :, 0] = new_actions
                        episode_sample['obs'][0, 1:, :, :] = new_obs
                        episode_sample['reward'][0, 0, 0] = rew
                        new_actions_onehot = th.zeros(episode_sample['actions'].squeeze(3).shape + (args.n_actions,))
                        new_actions_onehot = new_actions_onehot.scatter_(3, episode_sample['actions'].cpu(), 1)
                        episode_sample['actions_onehot'][:] = new_actions_onehot

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)

                        #print("action pair: %d, %d" % (i, j))
                        learner.train(episode_sample, runner.t_env, episode, show_demo=True, save_data=(i, j))
                last_demo_T = runner.t_env
                #time.sleep(1)
        #----------------------------------------------------------------------------------------------------

        #if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0 ):
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0 or runner.t_env > args.t_max):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", '-'.join(args.env_args['map_name']), args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)

            if args.git_root in ["QPLEX"]:
                if args.double_q:
                    os.makedirs(save_path + '_x', exist_ok=True)               

            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)
            # modify because of MAVEN
            # AttributeError: 'EpisodeRunner' object has no attribute 'save_models'
            if args.git_root in ["MAVEN"]:
                runner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    #modify
    if args.git_root in ["QPLEX"]:
        if args.is_save_buffer and save_buffer.is_from_start:
            save_buffer.is_from_start = False
            save_one_buffer(args, save_buffer, env_name, from_start=True)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config

def get_scheme_groups_preprocess(args,env_info):
    if args.git_root in ["AIQMIX"]:
        if not args.entity_scheme:
            # Default/Base scheme
            scheme = {
                "state": {"vshape": env_info["state_shape"]},
                "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
                "reward": {"vshape": (1,)},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
            groups = {
                "agents": args.n_agents
            }
        else:
            # Entity scheme
            scheme = {
                "entities": {"vshape": env_info["entity_shape"], "group": "entities"},
                "obs_mask": {"vshape": env_info["n_entities"], "group": "agents", "dtype": th.uint8},
                "entity_mask": {"vshape": env_info["n_entities"], "dtype": th.uint8},
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
                "reward": {"vshape": (1,)},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
            groups = {
                "agents": args.n_agents,
                "entities": args.n_entities
            }

        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
        }
    else:

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
        }
        #----------------------------------------
        if args.git_root in ["CollaQ"]:
            yy = {
                # This is added for Q_alone
                "obs_alone": {"vshape": env_info["obs_alone_shape"], "group": "agents"},
            }
            scheme.update(yy)
        #----------------------------------------
        if args.git_root in ["RODE"]:
            yy = {
                "role_avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
                "roles": {"vshape": (1,), "group": "agents", "dtype": th.long},
            }
            scheme.update(yy)
        #----------------------------------------
        if args.git_root in ["MAVEN"]:
            y = {
                "noise": {"vshape": (args.noise_dim,)}
            }
            scheme.update(y)


    return scheme,groups,preprocess


def process_batch(batch, args):

    if batch.device != args.device:
        batch.to(args.device)
    return batch

    # 完成合并
#  - ASN-------[✓]  
#  - AIQMIX----[✓]
#  - LICA------[✓]
#  - MAVEN-----[✓]
#  - WQMIX-----[✓]
#  - QTRAN-----[x]
#  - QPLEX-----[✓]  难
#  - CollaQ----[✓]
#  - ROMA------[✓]
#  - NDQ-------[✓]
#  - RODE------[✓]
#  - DOP-------[✓]
