def get_content_list():
    content_list = []
    env_list = [
        "sc2",
        # "gf"
        ]
    gpus = "0,1,2"
    git_root = "ROMA"

    algorithms_dict = {
        # "ROMA1":"qmix_smac_latent_gf",   #for gf
        "ROMA":"qmix_smac_latent",      #for sc
        "QMIX":"qmix_smac",
        "VDN":"vdn_smac",
        "IQL":"iql_smac",
    }
    map_str = "2s3z,5m_vs_6m,MMM2,8m_vs_9m,10m_vs_11m,2c_vs_64zg,27m_vs_30m"
    maps_list = map_str.split(",")
    map_special_dict = {
        "6z4b":"bane_vs_bane1",
        "10z5b_vs_2s3z":"zb_vs_sz",
        "6s4z_vs_10b30z":"sz_vs_zb",
    }
    for m_k in map_special_dict.keys():
        maps_list.append(map_special_dict.get(m_k))
   #------------------------------
    for env_name in env_list:
        env_conf=env_name
        content_list.append(f"## {env_conf}")
        for algorithm in algorithms_dict.keys():
            content_list.append("\n----------------------------\n")
            content_list.append(f"### {algorithm}")
            conf = algorithms_dict.get(algorithm)
            if env_name == "gf":
                conf += "_gf"
            for map_name in maps_list:
                content_list.append("### [ ] 完成(x   ✓)")
                if map_name == "MMM2":
                    cmd=f'python3 src/main.py --git-root={git_root} --config={conf} --env-config={env_conf} with agent=latent_ce_dis_rnn env_args.map_name={map_name}, t_max=20050000 use_tensorboard=True --gpus={gpus}'
                else:
                    cmd=f'python3 src/main.py --git-root={git_root} --config={conf} --env-config={env_conf} with agent=latent_ce_dis_rnn env_args.map_name={map_name}, t_max=20050000 h_loss_weight=5e-2 var_floor=1e-4 use_tensorboard=True --gpus={gpus}'

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