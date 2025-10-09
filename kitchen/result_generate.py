# %%
import torch
import pickle
import os
import numpy as np
import tqdm.notebook as tqdm
import copy

# %%

scenario_data = []
for i in range(300):
    pickle_file_path = "cache/data{}.pkl".format(i)
    if not os.path.exists(pickle_file_path):
        continue
    pickle_file = open(pickle_file_path, "rb")
    data = pickle.load(pickle_file)
    pickle_file.close()
    scenario_data.append(data)
print(f'加载了{len(scenario_data)}个场景数据')

# %%
def action_reformat(action):
    return action.replace('.', '')

# 数据处理 将动作对齐到answer index
def align_atomic_actions(data):
    print('_'*20, '开始处理新情景', '_'*20)
    ori_data = data
    data = copy.deepcopy(data)
    for k,v in data.items():
        if k == 'top_k_output' or k == 'conformal_prediction_output':
            for output_dict in v:
                top_k_result = output_dict['all_result']
                for k2, v2 in top_k_result.items():
                    action = v2['action']
                    for k3, v3 in data['atomic_dict'].items():
                        if k3 == action_reformat(action):
                            v2['action'] = v3['answer']
        if 'output' in k and 'new_' not in k:
            for output_dict in v:
                action = output_dict['action']
                for k2, v2 in data['atomic_dict'].items():
                    if k2 == action_reformat(action):
                        output_dict['action'] = v2['answer']
            ori_data['new_'+k] = v
            print(v)

# for data in scenario_data:
#     align_atomic_actions(data)
#     print('_'*20, '处理完成', '_'*20)
#     print(scenario_data)
# 未聚合情况下各方案正确答案提取
def evaluate_output_confidence1(data):
    data['result'] = {}
    for k,v in data.items():
        if 'output' in k and 'new_' in k:
            experiment_name = k.split('_')[1]
            data['result'][experiment_name] = {
                    'answer': data[k][0]['action'],
                    'confidence': data[k][0]['confidence'],
                    'right': data['right_answer'] == data[k][0]['action']
                }
            print(f'{experiment_name}: {data[k][0]["action"]}, {data[k][0]["confidence"]}, {data["right_answer"] == data[k][0]["action"]}')

# i = 299
# pickle_file_path = "cache/data{}.pkl".format(i)
# with open(pickle_file_path, "rb") as f:
#     data = pickle.load(f)
# align_atomic_actions(data)

# %%
def get_experiment_name(k):
    return k.replace('new_', '').replace('_output','')

# %%
# 未聚合情况下各方案正确答案提取
def evaluate_output_confidence1(data):
    data['result'] = {}
    for k,v in data.items():
        if 'output' in k and 'new_' in k:
            experiment_name = get_experiment_name(k)
            data['result'][experiment_name] = {
                    'answer': data[k][0]['action'],
                    'confidence': data[k][0]['confidence'],
                    'right': data['right_answer'] == data[k][0]['action']
                }
            print(f'{experiment_name}: {data[k][0]["action"]}, {data[k][0]["confidence"]}, {data["right_answer"] == data[k][0]["action"]}')

# %%
# 未聚合情况下各方案正确答案提取
import json
def evaluate_output_confidence_from_ambigous(data):
    index = data['index']
    experiment_name = 'ambigous'
    with open(f'content/task_data_llama/{index}.json', 'r') as f:
        task_data = json.load(f)
    confidence = 1 - task_data['ambiguous'] * 0.5
    data['result'][experiment_name] = {
            'answer': data['new_model0_output'][0]['action'],
            'confidence': confidence,
            'right': data['right_answer'] == data['new_model0_output'][0]['action']
        }
    confidence2 = confidence * data['new_model0_output'][0]['confidence']
    data['result']['ambigous_model0'] = {
            'answer': data['new_model0_output'][0]['action'],
            'confidence': confidence2,
            'right': data['right_answer'] == data['new_model0_output'][0]['action']
        }
    confidece3 = confidence * data['new_conformal_output'][0]['confidence']
    data['result']['ambigous_conformal'] = {
            'answer': data['new_conformal_output'][0]['action'],
            'confidence': confidece3,
            'right': data['right_answer'] == data['new_conformal_output'][0]['action']
        }
    
evaluate_output_confidence_from_ambigous(scenario_data[0])
scenario_data[0]['result']['ambigous_conformal']

# %%
for data in tqdm.tqdm(scenario_data):
    align_atomic_actions(data)
    if 'result' in data:
        del data['result']
    evaluate_output_confidence1(data)
    # generate_consistency_results(consistency_aggregation, data)
    # generate_avg_confidence_results(avg_confidence_aggregation, data)
    # generate_pair_rank_results(pair_rank_aggregation, data)
    evaluate_output_confidence_from_ambigous(data)




# %%

# 计算最终结果
class ListDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = []
        return super().__getitem__(key)

confidence_with_right = ListDict()
confidence_without_right = ListDict()
model_bias_record = ListDict()
for data in scenario_data:
    if 'result' not in data:
        continue
    print(data['result'])
    for experiment_name, result in data['result'].items():
        print(experiment_name)
        bias = result['confidence'] if not result['right'] else 1 - result['confidence']
        model_bias_record[experiment_name].append({
            'index': data['index'],
            'bias': bias
        })
        if result['right']:
            if experiment_name not in confidence_with_right:
                confidence_with_right[experiment_name] = []
            confidence_with_right[experiment_name].append(result['confidence'])
        else:
            if experiment_name not in confidence_without_right:
                confidence_without_right[experiment_name] = []
            confidence_without_right[experiment_name].append(result['confidence'])



# help rate v.s. success rate
result_with_confidence = ListDict()
for data in scenario_data:
    if 'result' not in data:
        continue
    for experiment_name, result in data['result'].items():
        result_with_confidence[experiment_name].append((result['confidence'], result['right']))
with open('pickle/result_with_confidence.pkl', 'wb') as f:
    pickle.dump(result_with_confidence, f)

# 对所有结果进行排序
for experiment_name in result_with_confidence:
    result_with_confidence[experiment_name].sort(key=lambda x: x[0])

# 不同Help Rate下的Success Rate
success_rate_conditioned_on_confidence = ListDict()
for experiment_name in result_with_confidence:
    for hr_percent in range(0, 101, 1):
        success_cache = []
        for ii, res_conf in enumerate(result_with_confidence[experiment_name]):
            if ii < len(result_with_confidence[experiment_name]) * hr_percent / 100:
                success_cache.append(1)
            elif res_conf[1]:
                success_cache.append(1)
            else:
                success_cache.append(0)
        if len(success_cache) > 0:
            success_rate_conditioned_on_confidence[experiment_name].append(np.mean(success_cache))
        else:
            success_rate_conditioned_on_confidence[experiment_name].append(1)
# 画图保存到img文件夹下
import matplotlib.pyplot as plt
import os
for experiment_name in success_rate_conditioned_on_confidence:
    plt.plot(success_rate_conditioned_on_confidence[experiment_name], label=experiment_name)
    # plot a line from (0, data[0]) to (100, 1)
    plt.plot([0, 100], [success_rate_conditioned_on_confidence[experiment_name][0], 1], label='linear')
    plt.xlabel('Help Rate')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig(f'img/{experiment_name}.png')
    plt.clf()
success_rate_conditioned_on_confidence_for_plot = copy.deepcopy(success_rate_conditioned_on_confidence)
with open('pickle/success_rate_conditioned_on_confidence.pkl', 'wb') as f:
    pickle.dump(success_rate_conditioned_on_confidence_for_plot, f)
# 均一化 防止初始高成功率对结果的影响
for experiment_name in success_rate_conditioned_on_confidence:
    # 所有实验数据减去第一个数据和最后一个数据的直线
    
    success_rate_conditioned_on_confidence[experiment_name] = np.array(success_rate_conditioned_on_confidence[experiment_name])


    # 新算法
    data_copy = copy.deepcopy(success_rate_conditioned_on_confidence[experiment_name])
    minus_data = success_rate_conditioned_on_confidence[experiment_name][-1] - success_rate_conditioned_on_confidence[experiment_name][0]
    minus_data = np.linspace(success_rate_conditioned_on_confidence[experiment_name][0], 1, len(success_rate_conditioned_on_confidence[experiment_name]))
    
    # 所有数据 除以 （1-第一个数据）* 第一个数据
    normal_coff = (1-success_rate_conditioned_on_confidence[experiment_name][0]) * success_rate_conditioned_on_confidence[experiment_name][0]

    data_copy -= minus_data
    # normal_coff = normal_coff * 0.5 * len(success_rate_conditioned_on_confidence[experiment_name])
    data_copy /= normal_coff

        
    divider = 0.5 * (len(success_rate_conditioned_on_confidence[experiment_name]) - 1)
    sr_hr_area1 = np.trapz(data_copy, dx=1) / divider

    print(experiment_name, sr_hr_area1)

            


# %%



