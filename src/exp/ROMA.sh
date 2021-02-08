#!/usr/bin/env bash
## sc2

----------------------------

### ROMA
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=2s3z, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=5m_vs_6m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=MMM2, t_max=20050000 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=8m_vs_9m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=10m_vs_11m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=2c_vs_64zg, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=27m_vs_30m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2

----------------------------

### QMIX
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=2s3z, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=5m_vs_6m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=MMM2, t_max=20050000 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=8m_vs_9m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=10m_vs_11m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=2c_vs_64zg, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=27m_vs_30m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2

----------------------------

### VDN
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=2s3z, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=5m_vs_6m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=MMM2, t_max=20050000 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=8m_vs_9m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=10m_vs_11m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=2c_vs_64zg, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=27m_vs_30m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2

----------------------------

### IQL
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=2s3z, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=5m_vs_6m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=MMM2, t_max=20050000 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=8m_vs_9m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=10m_vs_11m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=2c_vs_64zg, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac --env-config=sc2 with agent=latent_ce_dis_rnn env_args.map_name=27m_vs_30m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
## gf

----------------------------

### ROMA
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=2s3z, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=5m_vs_6m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=MMM2, t_max=20050000 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=8m_vs_9m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=10m_vs_11m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=2c_vs_64zg, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_latent_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=27m_vs_30m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2

----------------------------

### QMIX
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=2s3z, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=5m_vs_6m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=MMM2, t_max=20050000 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=8m_vs_9m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=10m_vs_11m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=2c_vs_64zg, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=qmix_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=27m_vs_30m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2

----------------------------

### VDN
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=2s3z, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=5m_vs_6m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=MMM2, t_max=20050000 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=8m_vs_9m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=10m_vs_11m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=2c_vs_64zg, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=vdn_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=27m_vs_30m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2

----------------------------

### IQL
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=2s3z, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=5m_vs_6m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=MMM2, t_max=20050000 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=8m_vs_9m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=10m_vs_11m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=2c_vs_64zg, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
### [ ] 完成(x   ✓)
python3 src/main.py --git-root=ROMA --config=iql_smac_gf --env-config=gf with agent=latent_ce_dis_rnn env_args.map_name=27m_vs_30m, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus=0,1,2
