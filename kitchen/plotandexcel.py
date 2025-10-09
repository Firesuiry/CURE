# %%
from scipy.stats import spearmanr

import pickle
import numpy as np
class ListDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = []
        return super().__getitem__(key)

exp_name_dict = {
    "ambigous-model0": "CURE-Ambiguity",  # 表示结合了模糊度评估的FUAA算法
    "model0-rnd": "CURE",  # 表示结合了RND任务相似度评估的FUAA算法
    "ambigous-conformal": "KnowNo-Ambiguity",  # 表示结合了模糊度评估的Conformal算法
    "model0": "CURE w/o sim",  # 表示不包含RND模块的FUAA算法
    "introplan": "IntroPlan",
    "conformal": "KnowNo",
    "ambigous": "Ambiguity",
}

def experiment_name_change(experiment_name: str):
    experiment_name = experiment_name.replace('_', '-')
    if experiment_name in exp_name_dict:
        return exp_name_dict[experiment_name]
    return experiment_name

def dict_change_key(d: dict):
    new_dict = {}
    for k, v in d.items():
        new_dict[experiment_name_change(k)] = v
    d = new_dict
    return new_dict


# %% 计算肯德尔相关系数
import scipy.stats as stats
import numpy as np

with open('pickle/result_with_confidence.pkl', 'rb') as f:
    result_with_confidence = pickle.load(f)
with open('introplan/introplan_result_with_confidence.pkl', 'rb') as f:
    introplan_result_with_confidence = pickle.load(f)
result_with_confidence['introplan'] = introplan_result_with_confidence
with open('pickle/result-with-confidence-rnd0.pkl', 'rb') as f:
    result_with_confidence_model0 = pickle.load(f)
result_with_confidence['model0'] = result_with_confidence_model0
with open('pickle/result-with-confidence-rnd30.pkl', 'rb') as f:
    result_with_confidence_model0_rnd = pickle.load(f)
result_with_confidence['model0-rnd'] = result_with_confidence_model0_rnd
result_with_confidence = dict_change_key(result_with_confidence)

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
with open(f'pickle/success_rate_conditioned_on_confidence.pkl', 'rb') as f:
    sc1 = pickle.load(f)
with open(f'introplan/introplan_success_rate_conditioned_on_confidence.pkl', 'rb') as f:
    sc2 = pickle.load(f)
sc1['introplan'] = sc2
for k, v in sc1.items():
    new_k = k.replace('_', '-')
    success_rate_conditioned_on_confidence[new_k] = v
success_rate_conditioned_on_confidence = dict_change_key(success_rate_conditioned_on_confidence)

# 画图保存到img文件夹下
import matplotlib.pyplot as plt
import os
import copy
import math

# 确保img文件夹存在
if not os.path.exists('img'):
    os.makedirs('img')

for k,v in success_rate_conditioned_on_confidence.items():
    print(k, len(v))
# del success_rate_conditioned_on_confidence['CURE w/o sim']
# del success_rate_conditioned_on_confidence['CURE']
experiment_names = list(success_rate_conditioned_on_confidence.keys())
num_experiments = len(experiment_names)
# 计算子图布局
num_cols = 3  # 每行3个子图
num_rows = math.ceil(num_experiments / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True, sharey=True)
axes = axes.flatten()
x = np.linspace(0, 1, 101)
for i, experiment_name in enumerate(experiment_names):
    ax = axes[i]
    ax.plot(x, success_rate_conditioned_on_confidence[experiment_name], label=experiment_name)
    ax.plot([0, 1], [success_rate_conditioned_on_confidence[experiment_name][0], 1], label='random')
    ax.set_title(experiment_name)
    ax.set_xlabel('Help Rate')
    ax.set_ylabel('Success Rate')
    ax.legend()

# 隐藏多余的子图
for j in range(num_experiments, num_rows * num_cols):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(f'img/all.png', dpi=400)
plt.close()


sr_hr_dict = {}
# 均一化 防止初始高成功率对结果的影响
for experiment_name in success_rate_conditioned_on_confidence:
    # 所有实验数据减去第一个数据和最后一个数据的直线

    success_rate_conditioned_on_confidence[experiment_name] = np.array(
        success_rate_conditioned_on_confidence[experiment_name])

    data_copy = copy.deepcopy(success_rate_conditioned_on_confidence[experiment_name])
    minus_data = success_rate_conditioned_on_confidence[experiment_name][-1] - \
                 success_rate_conditioned_on_confidence[experiment_name][0]
    minus_data = np.linspace(0, minus_data, len(success_rate_conditioned_on_confidence[experiment_name]))
    data_copy -= minus_data

    # 所有数据 除以 （1-第一个数据）* 第一个数据
    normal_coff = (1 - success_rate_conditioned_on_confidence[experiment_name][0]) * \
                  success_rate_conditioned_on_confidence[experiment_name][0]
    data_copy /= normal_coff

    sr_hr_area0 = np.trapz(data_copy, dx=1)

    # 新算法
    data_copy = copy.deepcopy(success_rate_conditioned_on_confidence[experiment_name])
    minus_data = success_rate_conditioned_on_confidence[experiment_name][-1] - \
                 success_rate_conditioned_on_confidence[experiment_name][0]
    minus_data = np.linspace(success_rate_conditioned_on_confidence[experiment_name][0], 1,
                             len(success_rate_conditioned_on_confidence[experiment_name]))

    # 所有数据 除以 （1-第一个数据）* 第一个数据
    normal_coff = (1 - success_rate_conditioned_on_confidence[experiment_name][0]) * \
                  success_rate_conditioned_on_confidence[experiment_name][0]

    data_copy -= minus_data
    # normal_coff = normal_coff * 0.5 * len(success_rate_conditioned_on_confidence[experiment_name])
    data_copy /= normal_coff
    divider = 0.5 * (len(success_rate_conditioned_on_confidence[experiment_name]) - 1)
    sr_hr_area1 = np.trapz(data_copy, dx=1) / divider

    print(experiment_name, sr_hr_area1)
    sr_hr_dict[experiment_name] = sr_hr_area1

kendall_result = []
spearman_result = []
for experiment_name in result_with_confidence:
    xs = [d[0] for d in result_with_confidence[experiment_name]]
    ys = [d[1] for d in result_with_confidence[experiment_name]]

    stats_res = stats.kendalltau(xs, ys)
    spearman_res = stats.spearmanr(xs, ys)
    res = {
        'experiment_name': experiment_name,
        'spearman': float(spearman_res.correlation),
        'spearman_p_value': float(spearman_res.pvalue),
        'sr_hr': sr_hr_dict[experiment_name]
    }
    kendall_result.append(res)
print(kendall_result)
import pandas as pd
df = pd.DataFrame(kendall_result)
# save as xlsx
df.to_excel('result.xlsx')
# %%
