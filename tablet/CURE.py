# %% [markdown]
# # 成功率和不确定性分离
# # tablet 实验

# %%
TEST_FLAG = False  # This is the test flag. If you need to reproduce the experiment, please set it to False
cache_dir = r"/root/autodl-tmp/models/"

# %%
freeze_base_model = True
parallel_train = False
encode_before_train = True

# %%
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModel, AutoTokenizer
import torch
import tqdm.notebook as tqdm
import pickle
device = torch.device('cuda')
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
import numpy as np
np.random.seed(0)

# %% [markdown]
# 

# %%

# device2 = torch.device('cuda:2')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "voidful/Llama-3.2-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
base_model = AutoModel.from_pretrained(model_name,cache_dir=cache_dir, load_in_8bit=True)

# %%
tokenizer.eos_token
tokenizer.pad_token = ' '
tokenizer.pad_token_id = tokenizer.encode(' ')[1]

# %%
import logging
import os
import datetime
import time


# %%
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
            nn.Linear(base_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        ])

    def forward(self, cls_output):
        logits = self.regression_head(cls_output)
        return logits

# %%
import torch.optim as optim


encode_model = EncoderModel(base_model).to(device)
regression_head = RegressionHead().to(device)


lr = 5e-5
optimizer = optim.Adam(regression_head.parameters(), lr=lr)
batch_size = 64

# %%
# read dataset
prompt_base = '''
You are a human and there is a robot. The robot are in front of a table.
On the table there are these objects: blue block, yellow bowl, yellow block, green bowl, green block, blue bowl.
The robot are asked to {task}.
Then the robot put the {pick_obj} to the {relation} of the {target_obj}.
Does the robot do the right thing?
'''
import random
from idna import encode
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import hashlib
import json

def in_model0_output(model0_output, model_output):
    for d in model0_output:
        if d['action'] == model_output:
            return True
    return False

class RealDataset(Dataset):
    def __init__(self, tokenizer, load_file='./task/dataset_val.json'):
        self.tokenizer = tokenizer
        self.data = []
        dataset = json.load(open(load_file, 'r'))
        for data in dataset:
            task = data['instruction']
            ambiguous = 1 if data['ambiguous'] else 0
            true_goal = data['goal'] + [1]
            false_goal = data['false_goal'] + [0]
            for goal in [true_goal, false_goal]:
                pick_obj = goal[0]
                relation = goal[1]
                target_obj = goal[2]
                prompt = prompt_base.format(task=task, pick_obj=pick_obj, relation=relation, target_obj=target_obj)
                self.data.append((prompt, (goal[3], ambiguous)))
            if len(self.data) > 99 and TEST_FLAG:
                break
        print(f'load {len(self.data)} samples')
        # 计算max_length
        self.max_length = 0
        for d in tqdm.tqdm(self.data):
            encoding = self.tokenizer(
                d[0],
                return_tensors="pt",
            )
            self.max_length = max(self.max_length, encoding["input_ids"].shape[1]) + 5
            break
            

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        item = prompt
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        encoding = self.tokenizer(
            item,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float).to(device),
            "md5": prompt_hash,
        }
        
class EncodeDataset(Dataset):
    def __init__(self, dataset):
        self.data = []
        if dataset is None:
            return
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for d in tqdm.tqdm(dataloader):
            item = d
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            label = item['label']
            md5s = item['md5']
            all_cached = True
            no_cached_list = []
            for i in range(len(md5s)):
                md5 = md5s[i]
                file_path = 'cache/encode/' + md5 + '.pickle'
                if not os.path.exists(file_path):
                    all_cached = False
                    # print(f'{file_path} not exists')
                    no_cached_list.append(i)
                else:
                    # print(f'{file_path} exists')
                    with open(file_path, 'rb') as f:
                        cls_output = pickle.load(f)
                    self.data.append({
                        "cls_output": cls_output,
                        "label": label[i],
                    })

            # create a new input_ids and attention_mask for no cached
            if len(no_cached_list) == 0:
                continue
            input_ids = input_ids[no_cached_list]
            attention_mask = attention_mask[no_cached_list]
            label = label[no_cached_list]
            md5s = [md5s[i] for i in no_cached_list]
            # no grad
            with torch.no_grad():
                cls_output = encode_model(input_ids, attention_mask)
                cls_output.to('cpu')
                input_ids = attention_mask = None
                torch.cuda.empty_cache()
            for i in range(len(label)):
                self.data.append({
                    "cls_output": cls_output[i].cpu().numpy(),
                    "label": label[i],
                })
                md5 = md5s[i]
                file_path = 'cache/encode/' + md5 + '.pickle'
                with open(file_path, 'wb') as f:
                    pickle.dump(cls_output[i].cpu().numpy(), f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


val_data_file = './task/dataset_val.json'
train_data_file = './task/dataset_train.json'

val_dataset = RealDataset(tokenizer, load_file=val_data_file)
val_size = len(val_dataset)
print('val_size', val_size)

train_dataset = RealDataset(tokenizer, load_file=train_data_file)
train_size = len(train_dataset)
print('train_size', train_size)


encode_val_dataset = EncodeDataset(val_dataset)

val_dataloader = DataLoader(encode_val_dataset, batch_size=batch_size, shuffle=False)
encode_train_dataset = EncodeDataset(train_dataset)

train_dataloader = DataLoader(encode_train_dataset, batch_size=batch_size, shuffle=True)

# %%
sample_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)
def sample_val_test():
    for i, data in enumerate(sample_dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        cls_output = encode_model(input_ids, attention_mask=attention_mask).float()
        label = data['label'].to(device)
        logits = regression_head(cls_output)
        # loss = loss_fn(logits.squeeze(-1), label)
        # print('val loss:', loss.item())
        for ii in range(20):
            print('_'*20+f'sample{ii}'+'_'*20)
            print('prompt', tokenizer.decode(input_ids[ii]))
            print(f'label:{label[ii]} predict:{logits[ii]}')
            print('_'*100)
        break
sample_val_test()

# %%
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


# %%
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
    
    # 返回概率
    return count / total_combinations

# 示例列表
L1 = [1, 3, 5]
L2 = [2, 4, 6]

probability = calculate_probability_optimized(L1, L2)
print(f"Probability that a number from L1 is greater than a number from L2: {probability}")

def test_inference(dataloader=val_dataloader, regression_head=regression_head):  
    detail_result.clear()
    # evaluate the model with val_dataset
    total_loss = 0
    all_accuracy = 0
    total_samples = 0
    current_index = 0
    label_0_predict = []
    label_1_predict = []
    for batch in dataloader:
        cls_output = batch['cls_output'].to(device).float()
        labels = batch['label'].to(device)
        # 取消梯度回传
        with torch.no_grad():
            outputs = regression_head(cls_output)
            loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        # add detail result
        batch_result = (outputs[:,0].squeeze(-1) > 0.5).int() == labels[:,0].int()
        for i in range(len(batch_result)):
            detail_result.append({
                'predict': outputs[i][0].item(),
                'label': labels[i][0].item(),
                'result': batch_result[i].item(),
            })
        # calculate the accuracy
        total_samples += labels.size(0)
    for res in detail_result:
        if res['label'] == 0:
            label_0_predict.append(res['predict'])
        else:
            label_1_predict.append(res['predict'])
    probability = calculate_probability_optimized(label_1_predict, label_0_predict)
    print(f"Total Loss: {total_loss/total_samples} Accuracy: {probability}")
    logging.info(f"Total Loss: {total_loss/total_samples} Accuracy: {probability}")
    return total_loss/total_samples, probability
test_inference(val_dataloader)
# if val_dataloader != train_dataloader:
#     test_inference(train_dataloader)

# %%
len(train_dataset)

# %%
l2_lambda = 0.01
val_loss = 999999999
val_acc = 0
no_better_epoch = 0
train_size = len(train_dataset)
new_train_dataset = EncodeDataset(None)
new_test_dataset = EncodeDataset(None)
train_dataset_data = encode_train_dataset.data.copy()
random.shuffle(train_dataset_data)
new_train_dataset.data = train_dataset_data[:int(train_size * 0.8)]
new_test_dataset.data = train_dataset_data[int(train_size * 0.8):]
train_dataloader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(new_test_dataset, batch_size=batch_size, shuffle=False)
regression_head = RegressionHead().to(device)
optimizer = optim.Adam(regression_head.parameters(), lr=lr)
for epoch in tqdm.tnrange(20):  # 设置训练轮次
    regression_head.train()
    total_loss = 0
    all_accuracy = 0
    total_samples = 0
    lable_0_predict = []
    lable_1_predict = []
    for batch in train_dataloader:
        cls_output = batch['cls_output'].to(device).float()
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = regression_head(cls_output)
        loss = loss_fn(outputs.squeeze(-1).float(), labels.float())
        # l2_reg = 0
        # if encode_before_train:
        #     for param in regression_head.parameters():
        #         l2_reg += torch.norm(param, p=2)
        # loss += l2_lambda * l2_reg
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy = ((outputs[:,0].squeeze(-1) > 0.5).int() == labels[:,0]).sum().item()
        total_samples += labels.size(0)
        all_accuracy += accuracy
        # for i in range(len(outputs)):
        #     if labels[i].item() == 0:
        #         lable_0_predict.append(outputs[i].item())
        #     else:
        #         lable_1_predict.append(outputs[i].item())
    # probability = calculate_probability_optimized(lable_1_predict, lable_0_predict)

    print(f"Epoch {epoch + 1}, Loss: {total_loss / total_samples}, Accuracy: {all_accuracy / total_samples}")
    if train_dataloader != val_dataloader:
        new_val_loss, new_val_acc = test_inference(regression_head=regression_head, dataloader=test_dataloader)
    else:
        new_val_loss = val_loss
        new_val_acc = val_acc
    # if encode_before_train:
    #     # save regression_head
    #     torch.save(regression_head.state_dict(), f'models/regression_head{epoch}.pth')
    if new_val_loss < val_loss or new_val_acc > val_acc:
        save_type = ''
        if new_val_loss < val_loss:
            val_loss = new_val_loss 
            save_type += 'loss'
        if new_val_acc > val_acc:
            val_acc = new_val_acc
            save_type += 'acc'

        if 'acc' in save_type:
            torch.save(regression_head.state_dict(), f'models/regression_head_best_acc.pth')
        if 'loss' in save_type:
            torch.save(regression_head.state_dict(), f'models/regression_head_best_loss.pth')
        print(f'model saved {save_type}')
        no_better_epoch = 0
    else:
        no_better_epoch += 1
    # test_inference(val_dataloader, regression_head=regression_head)

# load best model
# regression_head.load_state_dict(torch.load('models/regression_head_best_acc.pth'))
# test val dataset
# test_inference(val_dataloader, regression_head=regression_head)


# %%
#load best model
regression_head.load_state_dict(torch.load('models/regression_head_best_acc.pth'))
#test val dataset
test_inference(val_dataloader, regression_head=regression_head)

# %%
# RND model init and train

class RND(nn.Module):
    def __init__(self, base_model):
        super(RND, self).__init__()
        self.target = nn.Sequential(*[
            nn.Linear(base_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        ])
        self.predictor = nn.Sequential(*[
            nn.Linear(base_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        ])
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, cls_output):
        target  = self.target(cls_output)
        predictor = self.predictor(cls_output)
        return target, predictor
    
rnd_model = RND(base_model).to(device)
lr = 5e-5
rnd_optimizer = optim.Adam(rnd_model.predictor.parameters(), lr=lr)
rnd_loss_fn = nn.MSELoss()

for epoch in range(10):  # 设置训练轮次
    rnd_model.train()
    total_loss = 0
    total_samples = 0
    for batch in train_dataloader:

        cls_output = batch['cls_output'].to(device).float()
        optimizer.zero_grad()
        outputs = regression_head(cls_output)

        target, predictor = rnd_model(cls_output)
        loss = rnd_loss_fn(target, predictor)
        loss.backward()
        rnd_optimizer.step()

        total_loss += loss.item()
        total_samples += cls_output.size(0)

    print(f"Epoch {epoch + 1}, Loss: {total_loss / total_samples} len: {total_samples}")

# test inference
rnd_model.eval()
# train loss
total_loss = 0
total_samples = 0
for batch in train_dataloader:
    cls_output = batch['cls_output'].to(device).float()
    target, predictor = rnd_model(cls_output)
    loss = rnd_loss_fn(target, predictor)
    total_loss += loss.item()
    total_samples += cls_output.size(0)
print(f"Train Total Loss: {total_loss/total_samples} len: {total_samples}")


# %%
# val loss
total_loss = 0
total_samples = 0
loss_range = [999,-999]
loss_list = []
for batch in val_dataloader:
    cls_output = batch['cls_output'].to(device).float()
    label = batch['label'].to(device)
    target, predictor = rnd_model(cls_output)
    loss = rnd_loss_fn(target, predictor)
    total_loss += loss.item()
    total_samples += cls_output.size(0)
    for i in range(cls_output.size(0)):
        single_loss = rnd_loss_fn(target[i].unsqueeze(0), predictor[i].unsqueeze(0)).item()
        loss_range[0] = min(loss_range[0], single_loss)
        loss_range[1] = max(loss_range[1], single_loss)
        loss_list.append(single_loss)
print(f"Val Total Loss: {total_loss/total_samples} len: {total_samples} loss_range: {loss_range}")
loss_list


# %%
# load the best model
regression_head.load_state_dict(torch.load('models/regression_head_best_acc.pth'))


# %%
# prompt_base = '''
# You are a human and there is a robot operating in an office kitchen. The robot are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
# On the counter, there is {scence}.
# You says :"{task}".
# Then the robot {action}.
# Does the robot do the right thing?
# '''
import hashlib
regression_head.eval()
def get_confidence(task, pick_obj, relation, target_obj, rnd=0, detail_flag=False):
    prompt = prompt_base.format(task=task, pick_obj=pick_obj, relation=relation, target_obj=target_obj)
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file_name = f'cache/encode/{prompt_hash}.pkl'
    if Path(cache_file_name).exists():
        with open(cache_file_name, 'rb') as f:
            cls_output = pickle.load(f)
            cls_output = torch.tensor(cls_output).float().to(device)
    else:
        # print(prompt)
        encoding = tokenizer(
            prompt,
            max_length=val_dataset.max_length,
            padding='max_length',
            return_tensors="pt",
        )

        cls_output = encode_model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device)).squeeze(0)
        with open(cache_file_name, 'wb') as f:
            pickle.dump(cls_output.to('cpu').numpy(), f)

    
    outputs = regression_head(torch.tensor(cls_output).float().to(device))

    result = (outputs[0] * (1 - outputs[1]*0.6)).item()
    if rnd:
        target, predictor = rnd_model(cls_output)
        result = result - rnd_loss_fn(target, predictor).item()*rnd
    if detail_flag:
        return {'confidence':result,'s_conf': outputs[0].item(), 's_ambi': outputs[1].item()}
    else:
        return result
    # return {'confidence':result,'s_conf': outputs[0].item(), 's_ambi': outputs[1].item()}
get_confidence('open the top drawer', pick_obj='pcik_obj', relation='relation', target_obj='target_obj', rnd=1)

# %%
import numpy as np
class ListDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = []
        return super().__getitem__(key)
def evaluate(scenario_data, return_all=False):


    confidence_with_right = ListDict()
    confidence_without_right = ListDict()
    model_bias_record = ListDict()
    for data in scenario_data:
        if 'result' not in data:
            continue
        # print(data['result'])
        for experiment_name, result in data['result'].items():
            # print(experiment_name)
            bias = result['confidence'] if not result['right'] else 1 - result['confidence']
            model_bias_record[experiment_name].append({
                'index': data['index'],
                'bias': bias
            })
            if result['right']:
                if experiment_name not in confidence_with_right:
                    confidence_with_right[experiment_name] = []
                confidence_with_right[experiment_name].append(float(result['confidence']))
            else:
                if experiment_name not in confidence_without_right:
                    confidence_without_right[experiment_name] = []
                confidence_without_right[experiment_name].append(float(result['confidence']))


# help rate v.s. success rate
    result_with_confidence = ListDict()
    for data in scenario_data:
        if 'result' not in data:
            continue
        for experiment_name, result in data['result'].items():
            result_with_confidence[experiment_name].append((float(result['confidence']), result['right']))
    
    with open('tablet_result/result_with_confidence.pkl', 'wb') as f:
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
    with open('tablet_result/success_rate_conditioned_on_confidence.pkl', 'wb') as f:
        pickle.dump(success_rate_conditioned_on_confidence, f)
# 均一化 防止初始高成功率对结果的影响
    for experiment_name in success_rate_conditioned_on_confidence:
        # 所有实验数据减去第一个数据和最后一个数据的直线
        success_rate_conditioned_on_confidence[experiment_name] = np.array(success_rate_conditioned_on_confidence[experiment_name])
        minus_data = success_rate_conditioned_on_confidence[experiment_name][-1] - success_rate_conditioned_on_confidence[experiment_name][0]
        minus_data = np.linspace(success_rate_conditioned_on_confidence[experiment_name][0], 1, len(success_rate_conditioned_on_confidence[experiment_name]))
        
        # 所有数据 除以 （1-第一个数据）* 第一个数据
        normal_coff = (1-success_rate_conditioned_on_confidence[experiment_name][0]) * success_rate_conditioned_on_confidence[experiment_name][0]

        success_rate_conditioned_on_confidence[experiment_name] -= minus_data
        # normal_coff = normal_coff * 0.5 * len(success_rate_conditioned_on_confidence[experiment_name])
        success_rate_conditioned_on_confidence[experiment_name] /= normal_coff
    # assert normal_coff > 0

# 面积总和计算
    sr_hr_area = {}
    divider = 0.5 * (len(success_rate_conditioned_on_confidence[experiment_name]) - 1)
    for experiment_name in success_rate_conditioned_on_confidence:
        sr_hr_area[experiment_name] = np.trapz(success_rate_conditioned_on_confidence[experiment_name], dx=1) / divider

    final_result = []
    all_experiments = set(confidence_with_right.keys()) | set(confidence_without_right.keys())
    for experiment_name in all_experiments:
        # print(f'{experiment_name} with right answer: {np.mean(confidence_with_right[experiment_name])}')
        # print(f'{experiment_name} without right answer: {np.mean(confidence_without_right[experiment_name])}')
        final_result.append({
        'experiment_name': experiment_name,
        'confidence_with_right': np.mean(confidence_with_right[experiment_name]),
        'confidence_without_right': np.mean(confidence_without_right[experiment_name]),
        'sr_hr_area': sr_hr_area.get(experiment_name, 0)
        })
        print(experiment_name)
        if 'model0' in experiment_name and not return_all:
            return final_result[-1]
    return final_result

# %%
# add test result to uncertainty dataset

import json, pickle, os
import tqdm.notebook as tqdm

# cot conformal
def process_method(data, rnd, action_model ='conformal'):

    ori_model_output = data[f'{action_model}_output'][0]
    pick_obj = ori_model_output['pick_obj']
    relation = ori_model_output['relation']
    target_obj = ori_model_output['target_obj']
    instruction = data['instruction']
    confidence = get_confidence(instruction, pick_obj, relation, target_obj, rnd=rnd)
    print(f'action: {instruction}\nconfidence: {confidence}')
    model_output = []
    model_output.append({
        'pick_obj': pick_obj,
        'relation': relation,
        'target_obj': target_obj,
        'confidence': confidence
    })
    data[f'model0_output_rnd{rnd}'] = model_output


def process_cached_data(data_index, rnd):
    pickle_path = f"./cache/data{data_index}.pkl"

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print('-'*20, f'开始处理新情景[{data["index"]}]', '-'*20)
        try:
            process_method(data, rnd=rnd)
            evaluate_output_confidence1(data)
            # save data to ./cache/data[index].pkl
            with open(f'./cache/data{data["index"]}.pkl', 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            raise e
            print('error:', e)

def get_experiment_name(k):
    return k.replace('new_', '').replace('_output','')
# 未聚合情况下各方案正确答案提取
def evaluate_output_confidence1(data):
    def same(s1, s2):
        return s1.lower().strip() == s2.lower().strip()
    data['result'] = {}
    for k,v in data.items():
        if 'output' in k:
            experiment_name = k.replace('_output','')
            data['goal']
            
            data['result'][experiment_name] = {
                    'pick_obj': data[k][0]['pick_obj'],
                    'relation': data[k][0]['relation'],
                    'target_obj': data[k][0]['target_obj'],
                    'confidence': data[k][0]['confidence'],
                    'right': same(data['goal'][0], data[k][0]['pick_obj']) and same(data['goal'][1], data[k][0]['relation']) and same(data['goal'][2], data[k][0]['target_obj'])
                }
            print(f'{experiment_name}: {data['result'][experiment_name]}, {data['goal']}')

rnd_res = {}
rnds = [0,30]
all_data = []
for i in tqdm.trange(301):
    if not os.path.exists(f'./cache/tablet/data{i}.pkl'):
        continue
    with open(f'./cache/tablet/data{i}.pkl', 'rb') as f:
        data = pickle.load(f)
    try:
        for rnd in rnds:
            process_method(data, rnd=rnd)
        evaluate_output_confidence1(data)
        all_data.append(data)
    except Exception as e:
        print('error in', i)
        print(e)
res=evaluate(all_data, return_all=True)
rnd_res = res

print(rnd_res)
print('最终结果')
for r in res:
    print(r['experiment_name'], r['sr_hr_area'])




# %%
