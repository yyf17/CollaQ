#   修改
exp_id = '0000'

# 对于交换智能体类型，我们设计地图3s1z_vs_16zg，1s3z_vs_16zg和2s2z_vs_16zg（s表示缠扰者，z表示狂热者，zg表示虫族）。我们将前两张地图用于训练，将第三张地图用于测试。

# 对于添加单元，我们使用27m_vs_30m进行训练，使用28m_vs_30m进行测试（m代表海军陆战队）。

# 对于删除单元，我们使用29m_vs_30m进行训练，使用28m_vs_30m进行测试。

# 我们将所有实验进行4次，并在所有图中绘制均值/标准差。
# 包
def get_content_list():
    content_list = []
    algorithms_dict = {
        "IQL":"iql",
        "VDN":"vdn",
        "QTRAN":"qtran",
        "QMIX":"qmix",
        "CollaQ":"qmix_interactive_reg",
        "CollaQ with Attn":"qmix_interactive_reg_attn",
    }

    #----------------------------
    stage = "# Table 1"
    content_list.append(stage)
    map_str = "5m_vs_6m,MMM2,8m_vs_9m,10m_vs_11m,2c_vs_64zg,27m_vs_30m"
    maps_list = map_str.split(",")
    for algorithm in algorithms_dict.keys():
        content_list.append("\n----------------------------\n")
        content_list.append(f"# {algorithm}")
        conf = algorithms_dict.get(algorithm)
        for map_name in maps_list:
            content_list.append("### [ ] 完成(x   ✓)")
            cmd = f"python3 src/main.py --git-root=CollaQ --config={conf} --env-config=sc2 with env_args.map_name={map_name}, use_tensorboard=True --gpus=0,1,2"
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