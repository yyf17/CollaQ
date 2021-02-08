#!/usr/bin/env bash
************************************
# stand
## 8m_8m

-----------------------------------

### IQL
#### [ ] 完成(x   ✓)  Vanilla
python3 src/main.py --git-root=ASN --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Attention
python3 src/main.py --git-root=ASN --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Entity-Attention
python3 src/main.py --git-root=ASN --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_entity_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Dueling
python3 src/main.py --git-root=ASN --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_dueling' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  ASN
python3 src/main.py --git-root=ASN --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='asn_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2

-----------------------------------

### VDN
#### [ ] 完成(x   ✓)  Vanilla
python3 src/main.py --git-root=ASN --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Attention
python3 src/main.py --git-root=ASN --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Entity-Attention
python3 src/main.py --git-root=ASN --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_entity_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Dueling
python3 src/main.py --git-root=ASN --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_dueling' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  ASN
python3 src/main.py --git-root=ASN --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='asn_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2

-----------------------------------

### QMIX
#### [ ] 完成(x   ✓)  Vanilla
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Attention
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Entity-Attention
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_entity_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Dueling
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='dense_rnn_dueling' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  ASN
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_ASN' agents_num=8 enemies_num=8 agent='asn_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
************************************
# -1-Paddings
## 8m_8m

-----------------------------------

### QMIX
#### [ ] 完成(x   ✓)  Vanilla
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_not_0' agents_num=8 enemies_num=8 agent='dense_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Attention
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_not_0' agents_num=8 enemies_num=8 agent='dense_rnn_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Entity-Attention
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_not_0' agents_num=8 enemies_num=8 agent='dense_rnn_entity_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Dueling
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_not_0' agents_num=8 enemies_num=8 agent='dense_rnn_dueling' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  ASN
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_not_0' agents_num=8 enemies_num=8 agent='asn_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
************************************
# 1-Paddings
## 8m_8m

-----------------------------------

### QMIX
#### [ ] 完成(x   ✓)  Vanilla
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_set_1' agents_num=8 enemies_num=8 agent='dense_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Attention
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_set_1' agents_num=8 enemies_num=8 agent='dense_rnn_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Entity-Attention
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_set_1' agents_num=8 enemies_num=8 agent='dense_rnn_entity_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [ ] 完成(x   ✓)  Dueling
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_set_1' agents_num=8 enemies_num=8 agent='dense_rnn_dueling' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
#### [x ] 完成(x   ✓)  ASN
python3 src/main.py --git-root=ASN --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m, env='sc2_set_1' agents_num=8 enemies_num=8 agent='asn_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus=0,1,2
