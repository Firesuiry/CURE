#!/usr/bin/env python
# coding: utf-8

# In[4]:
import os
TEST_FLAG = os.getenv('TEST_FLAG', '0') == '1'
cache_dir = os.getenv('CACHE_DIR', '0')
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


# In[11]:


# read dataset
# read dataset
prompt_base = '''
You are a human and there is a robot operating in an office kitchen. The robot are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
On the counter, there is {scene}.
You says :"{task}".
Then the robot {action}.
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
    def __init__(self, tokenizer, balance=True):
        self.tokenizer = tokenizer
        self.data = []
        p = Path('cache/')
        for pkl_file in p.glob('data*.pkl'):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            scene = data['scene']
            task = data['task']
            index = data['index']
            with open(f'content/task_data_llama/{index}.json', 'r') as f:
                task_data = json.load(f)
            ambiguous = task_data['ambiguous'] * 1.0
            action_label = []
            for k,v in data['answer_dict'].items():
                action = v['reconstructed_action']
                label = 0 if k != data['right_answer'] else 1
                if 'NULL' in action:
                    continue
                if not in_model0_output(data['new_conformal_output'], k):
                    continue
                action_label.append((action, label))
            if balance:
                # 均衡化数据 使得label 0 和 1的数量相等
                label_0 = [d for d in action_label if d[1] == 0]
                label_1 = [d for d in action_label if d[1] == 1]
                min_len = min(len(label_0), len(label_1))
                action_label = random.sample(label_0, min_len) + random.sample(label_1, min_len)
                if len(action_label) == 0:
                    if len(label_0) == 0:
                        action_label = random.sample(label_1, 1)
                    else:
                        action_label = random.sample(label_0, 1)
                for action, label in action_label:
                    print(scene, task, action, label)
                    print('='*10)
                    self.data.append((prompt_base.format(scene=scene, task=task, action=action), (label, ambiguous)))
            else:
                for action, label in action_label:
                    self.data.append((prompt_base.format(scene=scene, task=task, action=action), (label, ambiguous)))
        print(f'load {len(self.data)} samples')
        # 计算max_length
        self.max_length = 0
        for d in tqdm.tqdm(self.data):
            encoding = self.tokenizer(
                d[0],
                return_tensors="pt",
            )
            self.max_length = max(self.max_length, encoding["input_ids"].shape[1])
            

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
                if not os.path.exists('cache/encode/'):
                    os.makedirs('cache/encode/')
                file_path = 'cache/encode/' + md5 + '.pickle'
                with open(file_path, 'wb') as f:
                    pickle.dump(cls_output[i].cpu().numpy(), f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


encode_dataset_file_path = 'cache/encode_real_dataset.pkl'
real_dataset = RealDataset(tokenizer, balance=False)

encode_real_dataset = EncodeDataset(real_dataset)
val_dataset = encode_real_dataset


val_size = len(val_dataset)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f'val_size: {val_size}')


# In[12]:


import json
from torch.utils.data import Dataset, DataLoader
import random
class MyDataset(Dataset):
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open('dataset_generate/task/task_action2.json', 'r') as f:
            all_task_action = json.load(f)
        for task_action in all_task_action:
            scene = task_action['Scene']
            actions = task_action['Action']
            ambiguous = task_action['Ambiguous']
            # 均衡正负样本
            label_0_actions = []
            label_1_actions = []
            for action_prob in actions:
                action = action_prob[0]
                label = action_prob[1]
                if label == 0:
                    label_0_actions.append(action_prob)
                else:
                    label_1_actions.append(action_prob)
            min_len = min(len(label_0_actions), len(label_1_actions))
            # assert min_len > 0, task_action
            if min_len == 0:
                continue
            actions = random.sample(label_0_actions, min_len) + random.sample(label_1_actions, min_len)
            for action_prob in actions:
                action = action_prob[0]
                label = (action_prob[1], ambiguous*1.0)
                prompt = prompt_base.format(task=task_action['Task'], action=action, scene=scene)
                self.data.append((prompt, label))
        print('data size:', len(self.data))
        max_length = 0
        i = 0
        for prompt, label in self.data:
            i += 1
            if i % 1000 == 0:
                print('processing:', i)
            encoding = self.tokenizer(
                prompt,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            max_length = max(max_length, encoding["input_ids"].shape[1])
        self.max_length = max_length
        print('max_length:', max_length)

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



dataset = MyDataset(tokenizer)
encode_train_dataset = EncodeDataset(dataset)

train_dataset = encode_train_dataset
# with open('cache/encode_train_dataset.pkl', 'wb') as f:
#     pickle.dump(train_dataset, f)

train_size = len(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f'train_size: {train_size}')



# In[10]:


sample_dataloader = DataLoader(real_dataset, batch_size=20, shuffle=False)
def sample_val_test():
    for i, data in enumerate(sample_dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        cls_output = encode_model(input_ids, attention_mask=attention_mask)
        label = data['label'].to(device)
        logits = regression_head(cls_output)
        # loss = loss_fn(logits.squeeze(-1), label)
        # print('val loss:', loss.item())
        for ii in range(1):
            print('_'*20+f'sample{ii}'+'_'*20)
            print('prompt', tokenizer.decode(input_ids[ii]))
            print(f'label:{label[ii]} predict:{logits[ii]}')
            print('_'*100)
        break
sample_val_test()

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
        cls_output = batch['cls_output'].to(device)
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
    #logging.info(f"Total Loss: {total_loss/total_samples} Accuracy: {probability}")
    return total_loss/total_samples, probability
test_inference(val_dataloader)
# if val_dataloader != train_dataloader:
#     test_inference(train_dataloader)

# In[84]:


def train_regression_head(encode_before_train, device, RegressionHead, loss_fn, lr, batch_size, EncodeDataset, val_dataloader, train_dataset, test_inference, train_size_ratio=1, random_dataset_split=True, only_dataloader=False):
    l2_lambda = 0.01
    val_loss = 999999999
    val_acc = 0
    no_better_epoch = 0
    train_size = len(train_dataset)*train_size_ratio
    new_train_dataset = EncodeDataset(None)
    new_test_dataset = EncodeDataset(None)
    train_dataset_data = train_dataset.data.copy()
    if random_dataset_split:
        train_dataset_data = random.sample(train_dataset_data, int(train_size))
    else:
        # 随机从train_dataset_data中取一段连续的的长度为train_size的数据
        start_index = random.randint(0, len(train_dataset_data)-int(train_size))
        train_dataset_data = train_dataset_data[start_index:start_index+int(train_size)]
    random.shuffle(train_dataset_data)
    new_train_dataset.data = train_dataset_data[:int(train_size * 0.8)]
    new_test_dataset.data = train_dataset_data[int(train_size * 0.8):]
    train_dataloader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(new_test_dataset, batch_size=batch_size, shuffle=False)
    if only_dataloader:
        return regression_head, train_dataloader
    regression_head = RegressionHead().to(device)
    optimizer = optim.Adam(regression_head.parameters(), lr=lr)
    for epoch in range(2):  # 设置训练轮次
        regression_head.train()
        total_loss = 0
        all_accuracy = 0
        total_samples = 0
        lable_0_predict = []
        lable_1_predict = []
        for batch in train_dataloader:
            cls_output = batch['cls_output'].to(device)
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

        # print(f"Epoch {epoch + 1}, Loss: {total_loss / total_samples}, Accuracy: {all_accuracy / total_samples}")

        new_val_loss, new_val_acc = test_inference(regression_head=regression_head, dataloader=test_dataloader)

        if new_val_acc > val_acc:
            val_acc = new_val_acc
            torch.save(regression_head.state_dict(), f'models/regression_head_best_acc.pth')
            print(f'[{epoch}]model saved with acc: {val_acc}')

    regression_head.load_state_dict(torch.load('models/regression_head_best_acc.pth'))
    test_loss, test_acc =test_inference(val_dataloader, regression_head=regression_head)
    print(f'test loss: {test_loss} test acc: {test_acc}')
    return regression_head, train_dataloader

# regression_head = train_regression_head(encode_before_train, device, RegressionHead, loss_fn, lr, batch_size, EncodeDataset, val_dataloader, train_dataset, test_inference)


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
    
def train_rnd_model(device, base_model, regression_head, val_dataloader, train_dataloader, RND, epochs=3, lr=5e-5, grad_accumulation_flag=False):
    rnd_model = RND(base_model).to(device)
    rnd_optimizer = optim.Adam(rnd_model.predictor.parameters(), lr=lr)
    rnd_loss_fn = nn.MSELoss()

    for epoch in range(epochs):  # 设置训练轮次
        rnd_model.train()
        total_loss = 0
        total_samples = 0
        for batch in train_dataloader:
            cls_output = batch['cls_output'].to(device)
            if not grad_accumulation_flag:
                rnd_optimizer.zero_grad()

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
        cls_output = batch['cls_output'].to(device)
        target, predictor = rnd_model(cls_output)
        loss = rnd_loss_fn(target, predictor)
        total_loss += loss.item()
        total_samples += cls_output.size(0)
    print(f"Train Total Loss: {total_loss/total_samples} len: {total_samples}")
# val loss
    total_loss = 0
    total_samples = 0
    for batch in val_dataloader:
        cls_output = batch['cls_output'].to(device)
        target, predictor = rnd_model(cls_output)
        loss = rnd_loss_fn(target, predictor)
        total_loss += loss.item()
        total_samples += cls_output.size(0)
    print(f"Val Total Loss: {total_loss/total_samples} len: {total_samples}")
    return rnd_model

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

def get_confidence(action, scene, task, rnd=1, rnd_model=None, regression_head=None, encode_model=encode_model, tokenizer=tokenizer, **kwargs):
    if regression_head:
        regression_head.eval()
    prompt = prompt_base.format(task=task, action=action, scene=scene)
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file_name = f'cache/encode_cls_output/{prompt_hash}.pkl'
    if not os.path.exists('cache/encode_cls_output/'):
        os.makedirs('cache/encode_cls_output/')
    if Path(cache_file_name).exists():
        with open(cache_file_name, 'rb') as f:
            cls_output = pickle.load(f)
            cls_output = torch.tensor(cls_output).to(device)
    else:
        # print(prompt)
        encoding = tokenizer(
            prompt,
            max_length=real_dataset.max_length,
            padding='max_length',
            return_tensors="pt",
        )

        cls_output = encode_model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device)).squeeze(0)
        with open(cache_file_name, 'wb') as f:
            pickle.dump(cls_output.to('cpu').numpy(), f)

    if rnd:
        rnd_loss_fn = nn.MSELoss()
        target, predictor = rnd_model(cls_output)
        if rnd > 9999:
            result = - rnd_loss_fn(target, predictor).item()
        else:
            outputs = regression_head(cls_output)
            result = (outputs[0] * (1 - outputs[1]*0.6)).item()
            result = result - rnd_loss_fn(target, predictor).item()*rnd
    else:
        outputs = regression_head(cls_output)
        result = (outputs[0] * (1 - outputs[1]*0.6)).item()

    return result


# %%计算最终结果
import numpy as np
# from utils.auc_evaluate import evaluate

# add test result to uncertainty dataset

import json, pickle, os
import tqdm.notebook as tqdm

def process_method(data, **kwargs):
    scence = data['scene']
    task = data['task']
    action_model = 'conformal'
    cot_output = data[f'new_{action_model}_output']
    model_output = []
    # print('-'*20, f'process_method[{data["index"]}]', '-'*20)
    # print(f'scene: {scence}\ntask: {task}')
    for i in range(len(cot_output)):
        action = data[f'{action_model}_output'][i]['action']
        reconstructed_action = data['answer_dict'][cot_output[i]['action']]['reconstructed_action']
        confidence = -1
        for d in model_output:
            if d['action'] == action:
                confidence = d['confidence']
                break
        if confidence == -1:
            confidence = get_confidence(reconstructed_action, scence, task, **kwargs)
            # print(f'action: {reconstructed_action}\nconfidence: {confidence}')
        model_output.append({
            'action': action,
            'confidence': confidence
        })
    data['model0_output'] = model_output


def process_cached_data(data_index,**kwargs):
    pickle_path = f"./cache/data{data_index}.pkl"
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        # print('-'*20, f'开始处理新情景[{data["index"]}]', '-'*20)
        try:
            process_method(data, **kwargs)
            align_atomic_actions(data)
            evaluate_output_confidence1(data)
            # save data to ./cache/data[index].pkl
            if 'save' in kwargs and kwargs['save']:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(data, f)
            return data
        except Exception as e:
            raise e
            print('error:', e)


def action_reformat(action):
    return action.replace('.', '')
import copy
# 数据处理 将动作对齐到answer index
def align_atomic_actions(data):
    # print('_'*20, '开始处理新情景', '_'*20)
    ori_data = data
    data = copy.deepcopy(data)
    for k,v in data.items():
        if 'model0' not in k:
            continue
        if 'output' in k and 'new_' not in k:
            for output_dict in v:
                action = output_dict['action']
                for k2, v2 in data['atomic_dict'].items():
                    if k2 == action_reformat(action):
                        output_dict['action'] = v2['answer']
            ori_data['new_'+k] = v
            # print(v)

def get_experiment_name(k):
    return k.replace('new_', '').replace('_output','')
# 未聚合情况下各方案正确答案提取
def evaluate_output_confidence1(data):
    data['result'] = {}
    for k,v in data.items():
        if 'output' in k and 'new_' in k and 'model0' in k:
            experiment_name = get_experiment_name(k)
            data['result'][experiment_name] = {
                    'answer': data[k][0]['action'],
                    'confidence': data[k][0]['confidence'],
                    'right': data['right_answer'] == data[k][0]['action']
                }
            # print(f'{experiment_name}: {data[k][0]["action"]}, {data[k][0]["confidence"]}, {data["right_answer"] == data[k][0]["action"]}')

# all_data = []
# for data_index in tqdm.tqdm(range(301)):
# # for data_index in tqdm.tqdm(range(3,4)):
#     data = process_cached_data(data_index, rnd=0, regression_head=regression_head,encode_model=encode_model, tokenizer=tokenizer)
#     if data:
#         all_data.append(data)


# result = evaluate(all_data)
# print(result)

# %%
def evaluate0(scenario_data, return_all=False, save_result_with_confidence=False, rnd=0):
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
        if normal_coff != 0:
            success_rate_conditioned_on_confidence[experiment_name] /= normal_coff

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
        if 'model0' in experiment_name and not return_all:
            if save_result_with_confidence:
            # save result with confidence
                sh_result = final_result[-1]['sr_hr_area']*1000
                pickle_name = f'pickle/result-with-confidence-rnd{rnd}.pkl'
                with open(pickle_name, 'wb') as f:
                    pickle.dump(result_with_confidence[experiment_name], f)
            return final_result[-1]
    return final_result
# In[86]:
def test_params(**config):
    train_flag = False
    for rnd in config['rnds']:
        if rnd < 999:
            train_flag = True
    regression_head, train_dataloader = train_regression_head(encode_before_train, device, RegressionHead, loss_fn, config['lr'], batch_size, EncodeDataset, val_dataloader, train_dataset, test_inference, train_size_ratio=config['train_size_ratio'], random_dataset_split=config['random_dataset_split'], only_dataloader=not train_flag)
    all_result = []
    rnd_model = train_rnd_model(device, base_model, regression_head, val_dataloader, train_dataloader, RND, epochs=config['rnd_epochs'], lr=5e-5, grad_accumulation_flag=config['grad_accumulation_flag'])
    for rnd in config['rnds']:
        all_data = []
        for data_index in range(301):
            data = process_cached_data(data_index, rnd=rnd, regression_head=regression_head,encode_model=encode_model, tokenizer=tokenizer, rnd_model=rnd_model, save=rnd==0)
            if data:
                all_data.append(data)

        result = evaluate0(all_data,save_result_with_confidence=True, rnd=rnd)
        all_result.append({
            'rnd': rnd,
            **config,
            'sr_hr_area': result['sr_hr_area']
        })
    return all_result


# %%
# test_params(train_size_ratio=0.01, lr=5e-5, rnds=[0,1,10])
# %%
configs = []
# for train_size_ratio in [0.0003,0.001,0.003, 0.01,0.1,1]:
config_dict = {
    'train_size_ratio': [0.001,0.003, 0.01,0.03,0.1,0.3,1],
    'rnd_epochs': [10],
    'rnds': [[100]],
    'random_dataset_split': [True],
    # 'use_rnd_online':[True, False],
    'lr': [5e-5],
    'grad_accumulation_flag':[False]
}
# generate all config combinations
import itertools
keys = config_dict.keys()
values = config_dict.values()
for instance in itertools.product(*values):
    config = dict(zip(keys, instance))
    configs.append(config)
import json
print(json.dumps(configs, indent=1))

# %%

final_result = []
configs = configs
for config in tqdm.tqdm(configs):
    results = []
    res = test_params(**config)
    final_result.extend(res)
    


# %%
final_result_dict = {}
for res in final_result:
    key = f''
    for k, v in res.items():
        if k not in ['sr_hr_area', 'rnds']:
            key += f'{k}_{v}_'
    if key not in final_result_dict:
        final_result_dict[key] = []
    final_result_dict[key].append(res['sr_hr_area'])
final_result_list = []
already_process = []
for res in final_result:
    key = f''
    for k, v in res.items():
        if k not in ['sr_hr_area', 'rnds']:
            key += f'{k}_{v}_'
    if key not in already_process:
        already_process.append(key)
        final_result_list.append({
            **res,
            'sr_hr_area': np.mean(final_result_dict[key])
        })

import pandas as pd
df = pd.DataFrame(final_result_list)
df.to_csv('params_search.csv', index=True)
df
    
# %%

