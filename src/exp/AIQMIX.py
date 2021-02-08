#   修改
exp_id = '0000'

# 包
import os

def get_content_list():
    algorithms =[
        'imagine_qmix',
        'atten_qmix',
        'imagine_vdn',
        'atten_vdn'
    ]
    environments =[
        'firefighters',
        'sc2custom',
    ]
    firefighters =[
        '2-8a_2-8b_05', #train on 5% of the possible scenarios
        '2-8a_2-8b_25', #train on 25% of the possible scenarios
        '2-8a_2-8b_45', #train on 45% of the possible scenarios
        '2-8a_2-8b_65', #train on 65% of the possible scenarios
        '2-8a_2-8b_85', #train on 85% of the possible scenarios
        '2-8a_2-8b_sim_Q1', #train on the quartile of scenarios least similar to the testing ones
        '2-8a_2-8b_sim_Q2', #train on the quartile of scenarios 2nd least similar to the testing ones
        '2-8a_2-8b_sim_Q3', #train on the quartile of scenarios 2nd most similar to the testing ones
        '2-8a_2-8b_sim_Q4', #train on the quartile of scenarios most similar to the testing ones
    ]
    sc2custom =[
        '3-8sz_symmetric',
        '3-8MMM_symmetric' 
    ]

    content_list = []

    for algorithm in algorithms:
        line_ = f'\n\n# {algorithm} ----------------------'
        print(line_)
        content_list.append(line_)
        for environment in environments:
            scenario_set_name_list = None
            if environment == 'firefighters':
                scenario_set_name_list = firefighters
            if environment == "sc2custom":
                scenario_set_name_list = sc2custom
            line_ = f'\n# {algorithm} {environment}'
            print(line_)
            content_list.append(line_)
            for scenario_set_name in scenario_set_name_list:
                comment = f"# [ ] 完成(x   ✓) {algorithm}  {environment}  {scenario_set_name}"
                print(comment)
                content_list.append(comment)
                cmd = f"python3 src/main.py --git-root=AIQMIX --config={algorithm} --env-config={environment} with scenario={scenario_set_name} --gpus=0,1,2"
                print(cmd)
                content_list.append(cmd)

    return content_list


def content_list_2_bash_file(content_list):
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
    
