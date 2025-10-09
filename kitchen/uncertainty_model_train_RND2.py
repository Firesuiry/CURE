#!/usr/bin/env python
# coding: utf-8

# 用于生成一些测试用例 不做最终输出

# In[4]:
import os
TEST_FLAG = os.getenv('TEST_FLAG', '0') == '1'
cache_dir = "/root/autodl-tmp/models/"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if cache_dir == "0":
    cache_dir = None

freeze_base_model = True
parallel_train = False
encode_before_train = True


# In[5]:


from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import tqdm.notebook as tqdm
import pickle
import os
from pathlib import Path
device = torch.device('cuda')

# 设置随机种子
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
import numpy as np
np.random.seed(0)



# In[6]:


# device2 = torch.device('cuda:2')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "voidful/Llama-3.2-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
base_model = AutoModel.from_pretrained(model_name,cache_dir=cache_dir,load_in_8bit=True)
# base_model = None

# In[7]:
prompt_base = '''
You are a human and there is a robot operating in an office kitchen. The robot are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
On the counter, there is {scene}.
You says :"{task}".
Then the robot {action}.
Does the robot do the right thing?
'''

tokenizer.eos_token
tokenizer.pad_token = ' '
tokenizer.pad_token_id = tokenizer.encode(' ')[1]





# In[9]:


import torch
import torch.nn as nn


# 分体式模型
class EncoderModel(nn.Module):
    def __init__(self, base_model):
        super(EncoderModel, self).__init__()
        self.base_model = base_model
        if base_model is not None:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, -1, :]
    
class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        self.regression_head = nn.Sequential(*[
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        ])

    def forward(self, cls_output):
        cls_output = cls_output.to(torch.float32)
        logits = self.regression_head(cls_output)
        return logits


# In[10]:


import torch.optim as optim


encode_model = EncoderModel(base_model).to(device)
regression_head = RegressionHead().to(device)


lr = 5e-5
optimizer = optim.Adam(regression_head.parameters(), lr=lr)
batch_size = 64

# %% loss
import torch
import torch.nn.functional as F

def loss_fn(outputs, labels):
    # 分离输出和标签中的成功率和模糊性
    success_rate_outputs = outputs[:, 0]
    ambiguity_outputs = outputs[:, 1]
    success_labels = labels[:, 0]
    ambiguity_labels = labels[:, 1]

    # 初始化总损失
    total_loss = torch.tensor(0.0, device=outputs.device)

    # 处理模糊性标签为0的情况：计算成功率和模糊性的均方误差
    non_ambiguous_mask = (ambiguity_labels == 0)
    if non_ambiguous_mask.any():
        success_rate_loss = F.mse_loss(success_rate_outputs[non_ambiguous_mask], 
                                       success_labels[non_ambiguous_mask])
        ambiguity_loss = F.mse_loss(ambiguity_outputs[non_ambiguous_mask], 
                                    ambiguity_labels[non_ambiguous_mask])
        total_loss += success_rate_loss + ambiguity_loss

    # 处理模糊性标签为1的情况：只计算模糊性的均方误差
    ambiguous_mask = (ambiguity_labels == 1)
    if ambiguous_mask.any():
        ambiguity_loss = F.mse_loss(ambiguity_outputs[ambiguous_mask], 
                                    ambiguity_labels[ambiguous_mask])
        total_loss += ambiguity_loss

    return total_loss

# # 示例使用
# outputs = torch.rand(10, 2)  # 假设有10个样本，每个样本有两个输出
# labels = torch.randint(0, 2, (10, 2)).float()  # 假设有10个样本，每个样本有两个标签
# loss = loss_fn(outputs, labels)
# print(loss)


# In[14]:
# inference
detail_result = []
from bisect import bisect_left

def calculate_probability_optimized(L1, L2):
    # 对 L2 排序
    L2.sort()
    
    # 初始化计数器
    count = 0
    
    # 遍历 L1 的每个数字
    for num in L1:
        # 使用二分查找找到 L2 中小于 num 的数字个数
        count += bisect_left(L2, num)
    
    # 计算总的组合数
    total_combinations = len(L1) * len(L2)
    if total_combinations == 0:
        return 0
    
    # 返回概率
    return count / total_combinations

# 示例列表
L1 = [1, 3, 5]
L2 = [2, 4, 6]


regression_head.load_state_dict(torch.load('models/regression_head_503.pth', map_location=device))



# In[74]:


# RND model init and train


class RND(nn.Module):
    def __init__(self, base_model):
        super(RND, self).__init__()
        self.target = nn.Sequential(*[
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        ])
        self.predictor = nn.Sequential(*[
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        ])
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, cls_output):
        cls_output = cls_output.to(torch.float32)
        target  = self.target(cls_output)
        predictor = self.predictor(cls_output)
        return target, predictor
    
rnd_model = RND(base_model).to(device)
rnd_model.load_state_dict(torch.load('models/rnd_model.pth', map_location=device))


# rnd_model = train_rnd_model(device, base_model, regression_head, val_dataloader, train_dataloader, RND)




# In[75]:


# prompt_base = '''
# You are a human and there is a robot operating in an office kitchen. The robot are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
# On the counter, there is {scence}.
# You says :"{task}".
# Then the robot {action}.
# Does the robot do the right thing?
# '''
import hashlib

def get_confidence(action, scene, task, rnd=1, encode_model=encode_model, tokenizer=tokenizer, **kwargs):
    if regression_head:
        regression_head.eval()
    prompt = prompt_base.format(task=task, action=action, scene=scene)
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file_name = f'cache/encode_cls_output/{prompt_hash}.pkl'
    if Path(cache_file_name).exists():
        with open(cache_file_name, 'rb') as f:
            cls_output = pickle.load(f)
            cls_output = torch.tensor(cls_output).to(device)
    else:
        # print(prompt)
        encoding = tokenizer(
            prompt,
            max_length=123,
            padding='max_length',
            return_tensors="pt",
        )

        cls_output = encode_model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device)).squeeze(0)
        with open(cache_file_name, 'wb') as f:
            pickle.dump(cls_output.to('cpu').numpy(), f)
    outputs = regression_head(cls_output)
    result = (outputs[0] * (1 - outputs[1]*0.6)).item()
    if rnd:
        rnd_loss_fn = nn.MSELoss()
        target, predictor = rnd_model(cls_output)
        with open('confidence_detail.txt', 'a', encoding='utf-8') as f:
            f.write(f'-'*20+'\n')
            f.write(f'rnd:{rnd}\n')
            f.write(f'action:{action}\n')
            f.write(f'scene:{scene}\n')
            f.write(f'task:{task}\n')
            f.write(f'success:{outputs[0]}\n')
            f.write(f'amb:{outputs[1]}\n')
            rnd_loss_cache = rnd_loss_fn(target, predictor).item()*rnd
            f.write(f'rnd_loss_fn:{rnd_loss_cache}\n')
            result = result - rnd_loss_cache
            f.write(f'confidence:{result}\n')
    else:
        outputs = regression_head(cls_output)
        result = (outputs[0] * (1 - outputs[1]*0.6)).item()

    return result

get_confidence('open the top drawer', 'a cup', 'open the top drawer', rnd=1)
get_confidence('pick-up hamburger to user', 'an orange, an apple, and a hamburger', 'give me a hamburger.', rnd=30)
get_confidence('pick-up hamburger to bottom drawer', 'an orange, an apple, and a hamburger', 'give me a hamburger.', rnd=30)
get_confidence('pick-up hamburger to microwave', 'an orange, an apple, and a hamburger', 'heat the hamburger.', rnd=30)
