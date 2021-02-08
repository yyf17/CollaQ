#!/usr/bin/env bash
# didactic task hallway
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=join1 with env_args.n_agents=2 env_args.state_numbers=6,6 obs_last_action=False comm_embed_dim=3 c_beta=0.1 comm_beta=1e-2 comm_entropy_beta=0. batch_size_run=16 t_max=2e7 is_cur_mu=True is_rank_cut_mu=True runner="parallel_x" test_interval=100000 use_tensorboard=True --gpus=0,1,2
# NDQ on SC2 tasks
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=bane_vs_hM env_args.obs_all_health=False comm_embed_dim=3 c_beta=0.1 comm_beta=0.0001 comm_entropy_beta=0.0 batch_size_run=16 runner="parallel_x" use_tensorboard=True --gpus=0,1,2

------------------------------

## NDQ1
### [ x ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=2s3z, use_tensorboard=True --gpus=2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=5m_vs_6m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=MMM2, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=8m_vs_9m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=10m_vs_11m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=2c_vs_64zg, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=27m_vs_30m, use_tensorboard=True --gpus=0,1,2

------------------------------

## NDQ2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=tar_qmix --env-config=sc2 with env_args.map_name=2s3z, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=tar_qmix --env-config=sc2 with env_args.map_name=5m_vs_6m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=tar_qmix --env-config=sc2 with env_args.map_name=MMM2, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=tar_qmix --env-config=sc2 with env_args.map_name=8m_vs_9m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=tar_qmix --env-config=sc2 with env_args.map_name=10m_vs_11m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=tar_qmix --env-config=sc2 with env_args.map_name=2c_vs_64zg, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=tar_qmix --env-config=sc2 with env_args.map_name=27m_vs_30m, use_tensorboard=True --gpus=0,1,2

------------------------------

## QMIX
### [ x ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=qmix_smac --env-config=sc2 with env_args.map_name=2s3z, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=qmix_smac --env-config=sc2 with env_args.map_name=5m_vs_6m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=qmix_smac --env-config=sc2 with env_args.map_name=MMM2, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_vs_9m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=qmix_smac --env-config=sc2 with env_args.map_name=10m_vs_11m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=qmix_smac --env-config=sc2 with env_args.map_name=2c_vs_64zg, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=qmix_smac --env-config=sc2 with env_args.map_name=27m_vs_30m, use_tensorboard=True --gpus=0,1,2

------------------------------

## VDN
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=vdn_smac --env-config=sc2 with env_args.map_name=2s3z, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=vdn_smac --env-config=sc2 with env_args.map_name=5m_vs_6m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=vdn_smac --env-config=sc2 with env_args.map_name=MMM2, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_vs_9m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=vdn_smac --env-config=sc2 with env_args.map_name=10m_vs_11m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=vdn_smac --env-config=sc2 with env_args.map_name=2c_vs_64zg, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=vdn_smac --env-config=sc2 with env_args.map_name=27m_vs_30m, use_tensorboard=True --gpus=0,1,2

------------------------------

## IQL
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=iql_smac --env-config=sc2 with env_args.map_name=2s3z, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=iql_smac --env-config=sc2 with env_args.map_name=5m_vs_6m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=iql_smac --env-config=sc2 with env_args.map_name=MMM2, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=iql_smac --env-config=sc2 with env_args.map_name=8m_vs_9m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=iql_smac --env-config=sc2 with env_args.map_name=10m_vs_11m, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=iql_smac --env-config=sc2 with env_args.map_name=2c_vs_64zg, use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=NDQ --config=iql_smac --env-config=sc2 with env_args.map_name=27m_vs_30m, use_tensorboard=True --gpus=0,1,2
