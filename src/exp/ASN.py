
####  添加ASN: Action Semantics Network: Considering the Effects of Actions in Multiagent Systems
# 实验部分

content_list = []

agent_type_dict = {
    "Vanilla":'dense_rnn',
    "Attention":'dense_rnn_attention',
    "Entity-Attention":'dense_rnn_entity_attention',
    "Dueling":'dense_rnn_dueling' ,
    "ASN":'asn_rnn',
}

algorithms_dict = {
    "IQL":"iql_smac",
    "VDN":"vdn_smac",
    "QMIX":"qmix_smac",
}

map_dict = {
    '8m':"8m_8m",
}

env_dict = {
    "stand":"sc2_ASN",
    "-1-Paddings":'sc2_not_0',
    "1-Paddings":"sc2_set_1",
}

gpus="0,1,2"

def get_content_list():
    content_list = []
    #------------------------
    content_list.append("************************************")
    env_key = "stand"
    env = env_dict.get(env_key)
    env_config = "sc2"
    content_list.append(f"# {env_key}")
    map_name = map_dict["8m"]
    content_list.append(f"## {map_name}")
    for algorithm in algorithms_dict.keys():
        content_list.append("\n-----------------------------------\n")
        content_list.append(f"### {algorithm}")
        algorithm_config = algorithms_dict[algorithm]
        for agent in agent_type_dict.keys():
            content_list.append(f"#### [ ] 完成(x   ✓)  {agent}")
            agent_name = agent_type_dict.get(agent)
            cmd = f"python3 src/main.py --git-root=ASN --config={algorithm_config} --env-config={env_config} with env_args.map_name={map_name}, env=\'{env}\' agents_num=8 enemies_num=8 agent=\'{agent_name}\' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus={gpus}"
            print(cmd)
            content_list.append(cmd)

   #------------------------
    content_list.append("************************************")
    env_config = "sc2"
    env_key = "-1-Paddings"
    env = env_dict.get(env_key)
    content_list.append(f"# {env_key}")
    map_name = map_dict["8m"]
    content_list.append(f"## {map_name}")
    
    algorithm="QMIX"
    algorithm_config = algorithms_dict[algorithm]
    content_list.append("\n-----------------------------------\n")
    content_list.append(f"### {algorithm}")
    for agent in agent_type_dict.keys():
        content_list.append(f"#### [ ] 完成(x   ✓)  {agent}")
        agent_name = agent_type_dict.get(agent)
        cmd = f"python3 src/main.py --git-root=ASN --config={algorithm_config} --env-config={env_config} with env_args.map_name={map_name}, env=\'{env}\' agents_num=8 enemies_num=8 agent=\'{agent_name}\' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus={gpus}"
        print(cmd)
        content_list.append(cmd)

    #------------------------
    content_list.append("************************************")
    env_config = "sc2"
    env_key = "1-Paddings"
    env = env_dict.get(env_key)
    content_list.append(f"# {env_key}")
    map_name = map_dict["8m"]
    content_list.append(f"## {map_name}")
    
    algorithm="QMIX"
    algorithm_config = algorithms_dict[algorithm]
    content_list.append("\n-----------------------------------\n")
    content_list.append(f"### {algorithm}")
    for agent in agent_type_dict.keys():
        content_list.append(f"#### [ ] 完成(x   ✓)  {agent}")
        agent_name = agent_type_dict.get(agent)
        cmd = f"python3 src/main.py --git-root=ASN --config={algorithm_config} --env-config={env_config} with env_args.map_name={map_name}, env=\'{env}\' agents_num=8 enemies_num=8 agent=\'{agent_name}\' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 --gpus={gpus}"
        print(cmd)
        content_list.append(cmd)


    return content_list

def content_list_2_bash_file(content_list):
    import os
    #命令写入bash
    bash_name =  __file__
    bash_name = bash_name.replace('.py', '.sh')
    if os.path.exists(bash_name):
        os.remove(bash_name)

    with open(bash_name,'w') as f:

        f.write('#!/usr/bin/env bash\n')
        for cont in content_list:
            f.write(f"{cont}\n")
        f.close()
    # 修改命令文件的权限，保证能够执行命令
    os.chmod(bash_name, int('770',base=8))

if __name__ == '__main__':
    content_list = get_content_list()
    content_list_2_bash_file(content_list)