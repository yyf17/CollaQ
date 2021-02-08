def get_content_list():
    content_list = []
    env_list = ["sc2","tracker1","join1"]
    gpus = "0,1,2"

    algorithms_dict = {
        "NDQ1":"categorical_qmix",
        "NDQ2":"tar_qmix",
        "QMIX":"qmix_smac",
        "VDN":"vdn_smac",
        "IQL":"iql_smac",
    }
    map_str = "2s3z,5m_vs_6m,MMM2,8m_vs_9m,10m_vs_11m,2c_vs_64zg,27m_vs_30m"
    maps_list = map_str.split(",")
   #------------------------------
    stage="# didactic task hallway"
    content_list.append(stage)
    content_list.append("### [ ] 完成(x   ✓)")
    cmd=f'python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=join1 with env_args.n_agents=2 env_args.state_numbers=6,6 obs_last_action=False comm_embed_dim=3 c_beta=0.1 comm_beta=1e-2 comm_entropy_beta=0. batch_size_run=16 t_max=2e7 is_cur_mu=True is_rank_cut_mu=True runner="parallel_x" test_interval=100000 use_tensorboard=True --gpus={gpus}'
    content_list.append(cmd)

    stage="# NDQ on SC2 tasks"
    content_list.append(stage)
    content_list.append("### [ ] 完成(x   ✓)")
    #cmd=f'python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=bane_vs_hM env_args.sight_range=2 env_args.shoot_range=2 env_args.obs_all_health=False env_args.obs_enemy_health=False comm_embed_dim=3 c_beta=0.1 comm_beta=0.0001 comm_entropy_beta=0.0 batch_size_run=16 runner="parallel_x" --gpus={gpus}'
    cmd=f'python3 src/main.py --git-root=NDQ --config=categorical_qmix --env-config=sc2 with env_args.map_name=bane_vs_hM env_args.obs_all_health=False comm_embed_dim=3 c_beta=0.1 comm_beta=0.0001 comm_entropy_beta=0.0 batch_size_run=16 runner="parallel_x" use_tensorboard=True --gpus={gpus}'
    content_list.append(cmd)

    
    for algorithm in algorithms_dict.keys():
        content_list.append("\n------------------------------\n")
        content_list.append(f"## {algorithm}")
        conf = algorithms_dict.get(algorithm)
        for map_name in maps_list:
            content_list.append("### [ ] 完成(x   ✓)")
            cmd = f"python3 src/main.py --git-root=NDQ --config={conf} --env-config=sc2 with env_args.map_name={map_name}, use_tensorboard=True --gpus={gpus}"
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