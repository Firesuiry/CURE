# %%
import os
TEST_FLAG = os.getenv('TEST_FLAG', '0') == '1'
cache_dir = os.getenv('CACHE_DIR', '0')
if cache_dir == "0":
    cache_dir = "/root/autodl-tmp/models/"

# %%
#@markdown A few imports and downloading data
# !pip install -U --no-cache-dir gdown --pre
# !pip install openai tqdm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 设置可用显卡为0，2，3
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'

# import openai
import signal
import tqdm.notebook as tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import pdb
import pickle
import copy
# dataset files - small sizes
# !gdown https://drive.google.com/uc?id=1XWIeGfF08V1eR104VLDilmwhIGVk2uzk
# !gdown https://drive.google.com/uc?id=1iEIZaVbbajMXsNdrjVkOgK5rhPtfl5WI

# Set OpenAI API key.
# openai.api_key = openai_api_key

# %%
NUM_ENSEMBLE=1
debug = True

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor
from transformers import LogitsProcessorList, LogitsProcessor
import torch
from llm_llma import LLAMA

try:
    llama_obj
except:
    # model_name = "unsloth/Llama-3.3-70B-Instruct"
    model_name = "voidful/Llama-3.2-8B-Instruct"
    llama_obj = LLAMA(model_name, load_in_8bit=True)
    llama = llama_obj.llama

    tokenizer = llama_obj.tokenizer
    class RestrictTokenLogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer, allowed_tokens):
            self.allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)

        def __call__(self, input_ids, scores):
            # Set logits of all tokens except the allowed ones to -inf
            forbidden_tokens_mask = torch.ones_like(scores).bool()
            forbidden_tokens_mask[:, self.allowed_token_ids] = False
            scores[forbidden_tokens_mask] = float('-inf')
            return scores

    allowed_tokens = ['A', 'B', 'C', 'D', 'E']
    allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
    processors = LogitsProcessorList([
        RestrictTokenLogitsProcessor(tokenizer, allowed_tokens),
        InfNanRemoveLogitsProcessor()  # Removes inf/nan values to prevent errors during generation
    ])

    yes_no_processors = LogitsProcessorList([
        RestrictTokenLogitsProcessor(tokenizer, ['Yes', 'No']),
        InfNanRemoveLogitsProcessor()  # Removes inf/nan values to prevent errors during generation
    ])

# %%
import json
task_data_path = './task/dataset_val.json'
# 如果报错请先运行./task/tablet_task_generate.ipynb
with open(task_data_path, 'r') as f:
  task_dataset = json.load(f)
task_dataset[0]
if TEST_FLAG:
    task_dataset = task_dataset[:10]

# %%
all_scence = '''You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.
'''.replace('\n', '')

action_example = '''
put blue bowl at the right side of blue block
put green bowl at the right side of blue bowl
put green block at the right side of blue bowl
put yellow bowl at the right side of blue bowl
put yellow block in front of green block
put yellow block behind green block
put yellow block to the left of green block
put yellow block to the right of green block
put yellow bowl at the front of the green block
put yellow bowl at the left side of the green block
put yellow bowl at the left side of the blue block
put yellow bowl at the right side of the green block
'''


# %%
tokenizer = llama_obj.tokenizer
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def chat(messages, max_length=1000, temperature=0.0, do_sample=True):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llama_obj.device)
    decode_outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    outputs = llama_obj.model.generate(
        input_ids,
        max_new_tokens=max_length,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    messages.append({
        'role': 'assistant',
        'content': response
    })
    return response, messages

def chat_with_score(messages, max_length=1, allowed_token_ids=['Yes', 'No']):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llama_obj.device)
    logits_processor = LogitsProcessorList([
    RestrictTokenLogitsProcessor(tokenizer, allowed_token_ids),
    InfNanRemoveLogitsProcessor()  # Removes inf/nan values to prevent errors during generation
    ])
    outputs = llama_obj.model.generate(
        input_ids,
        max_new_tokens=max_length,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor,
        output_scores=True,
        return_dict_in_generate=True, 
    )
    last_token_logits = outputs.scores[-1]
    last_token_logits = last_token_logits.detach().cpu()
    probs = torch.softmax(last_token_logits, dim=-1)

    all_tokens = ['Yes', 'No']
    allowed_token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    token_probs = {}
    for j in range(len(all_tokens)):
        log_prob = probs[0, allowed_token_ids[j]].item()
        token_probs[all_tokens[j]] = log_prob
    return token_probs

messages = [
    {
        'role': 'system',
        'content': 'You are a assistent, please reply according to the user\'s request.'
    },
    {
        'role': 'user',
        'content': '1+1=2?say Yes or No'
    }
]
print(chat_with_score(messages, max_length=1, allowed_token_ids=['Yes', 'No']))

# %%
# Vanilla 方法输出不确定性
vanilla_prompt = '''
provide what will you do and your confidence in this answer. 
Note: The confidence indicates how likely you think what will you do is true. 
Please provide the answer in the following format:
"Pick object:[the object you will pick(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]
Relation:[the relation between the object you will pick and the object you will put it(should be one of 'in' 'front', 'back', 'left', 'right')]
Target_obj:[the object you will put it(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]
Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%"
Only the answer and confidence, don't give me the explanation.  
Now, please answer this question and provide your confidence level.
'''
{'index': 0,
 'environment': ['green block',
  'blue block',
  'yellow block',
  'green bowl',
  'blue bowl',
  'yellow bowl'],
 'instruction': 'put the block in the green bowl',
 'goal': ['green block', 'in', 'green bowl']}
def prompt_vanilla(data, sample_num=1, sample_mode='greedy', print_flag = True):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=','.join(data["environment"]))},
        {"role": "user", "content": "{task}".format(task=data["instruction"])},
        {"role": "user", "content": vanilla_prompt},
    ]
    response, messages = chat(messages, max_length=1000)

    if print_flag:print(response)
    for line in response.split('\n'):
        if 'Confidence:' in line:
            output_confidence = float(line.split('Confidence:')[1].strip().replace('%', ''))/100
        if 'Pick object:' in line:
            pick_obj = line.split('Pick object:')[1].strip()
        if 'Relation:' in line:
            relation = line.split('Relation:')[1].strip()
        if 'Target_obj:' in line:
            target_obj = line.split('Target_obj:')[1].strip()

    
    vanilla_output = {
        'pick_obj': pick_obj,
        'relation': relation,
        'target_obj': target_obj,
        'confidence': output_confidence
    }
    return vanilla_output
def collect_vanilla_outputs(prompt_vanilla, data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'vanilla_output' in data and not force_rerun:
        return
    all_outputs = []
    for i in range(num_ensemble):
        vanilla_output = prompt_vanilla(data)
        all_outputs.append(vanilla_output)
    print(f'task: {data["instruction"]}\nvanilla_output: {vanilla_output}\n')
    print('-'*100)
    data['vanilla_output'] = all_outputs
    return data
    


data = task_dataset[128]
collect_vanilla_outputs(prompt_vanilla, data, force_rerun=True, num_ensemble=1)


# %%
# COT 方法输出不确定性
cot_prompt = '''
analyze step by step, provide your what will you do and your confidence in this answer. 
'''
cot_prompt = '''
analyze step by step, provide what will you do and your confidence in this answer. 
Note: The confidence indicates how likely you think your action is true. Use the following format to answer: 
"Explanation: [insert step-by-step analysis here] 
Pick object:[the object you will pick(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]
Relation:[the relation between the object you will pick and the object you will put it(should be one of 'in' 'front', 'back', 'left', 'right')]
Target_obj:[the object you will put it(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]
Confidence:[Your confidence level, please only include the numerical number in the range of 0-100]%"
Note: The action you take should be one of [pick-up, bring, move, open, close] and something you seen(should be a single full name) [to where(should be a single full name) you seen if needed].
Only give me the reply according to this format, don't give me any other words. 
Now, please answer this question and provide your confidence level. Let's think it step by step.
'''
def prompt_cot(data, sample_num=1, sample_mode='greedy'):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=','.join(data["environment"]))},
        {"role": "user", "content": "{task}".format(task=data["instruction"])},
        {"role": "user", "content": cot_prompt},
    ]
    response, messages = chat(messages, max_length=1000)
    messages_for_confidence = copy.deepcopy(messages)
    print(response)
    for line in response.split('\n'):
        if 'Pick object:' in line:
            pick_obj = line.split('Pick object:')[1].strip()
        if 'Relation:' in line:
            relation = line.split('Relation:')[1].strip()
        if 'Target_obj:' in line:
            target_obj = line.split('Target_obj:')[1].strip()
        if 'Confidence:' in line:
            output_confidence = float(line.split('Confidence:')[1].strip().replace('%', ''))/100
    
    cot_output = {
        'pick_obj': pick_obj,
        'relation': relation,
        'target_obj': target_obj,
        'confidence': output_confidence
    }
    return cot_output


def collect_cot_output(data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'cot_output' in data and not force_rerun:
        return
    all_outputs = []
    for i in range(num_ensemble):
        cot_output = prompt_cot(data)
        all_outputs.append(cot_output)
    print(f'task: {data["instruction"]}\ncot_output: {cot_output}\n')
    print('-'*100)
    data['cot_output'] = all_outputs
    return data['cot_output']

data = task_dataset[128]
collect_cot_output(data, force_rerun=True, num_ensemble=1)

# %%
self_probing_prompt = '''
What will you do? 
Use the following format to answer: 
"Pick object:[the object you will pick(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]
Relation:[the relation between the object you will pick and the object you will put it(should be one of 'in' 'front', 'back', 'left', 'right')]
Target_obj:[the object you will put it(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]"
'''
def prompt_self_probing(data, sample_num=1, sample_mode='greedy'):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=','.join(data["environment"]))},
        {"role": "user", "content": "{task}".format(task=data["instruction"])},
        {"role": "user", "content": self_probing_prompt},
    ]
    print(messages)
    response, messages = chat(messages, max_length=1000)
    messages_for_confidence = copy.deepcopy(messages)
    messages_for_confidence2 = copy.deepcopy(messages)
    print(response)
    for line in response.split('\n'):
        if 'Pick object:' in line:
            pick_obj = line.split('Pick object:')[1].strip()
        if 'Relation:' in line:
            relation = line.split('Relation:')[1].strip()
        if 'Target_obj:' in line:
            target_obj = line.split('Target_obj:')[1].strip()
    messages_for_confidence.append({
        'role': 'user',
        'content': 'according to your response, what is your confidence in this action? reply with the confidence only. the confidence should be a percentage.'
    })
    response, messages_for_confidence = chat(messages_for_confidence, max_length=1000)
    print('*'*100)
    print(response)
    output_confidence = float(response.split('%')[0])/100
    cot_output = {
        'pick_obj': pick_obj,
        'relation': relation,
        'target_obj': target_obj,
        'confidence': output_confidence
    }
    # confidence collect with yes/no logits
    messages_for_confidence2.append({
        'role': 'user',
        'content': 'according to your response, do you think your action is correct? reply with Yes or No.'
    })
    res = chat_with_score(messages_for_confidence2, max_length=1, allowed_token_ids=['Yes', 'No'])
    print(res)
    confidence = res['Yes']
    print(f'log confidence: {confidence}')
    cot_output['log_confidence'] = confidence
    return cot_output

def collect_self_prob_outputs(data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'self_probing_output' in data and not force_rerun:
        return
    all_outputs = []
    all_outputs_log = []
    for i in range(num_ensemble):
        self_probing_output = prompt_self_probing(data)
        self_probing_output_log = copy.deepcopy(self_probing_output)

        del self_probing_output['log_confidence']
        all_outputs.append(self_probing_output)
        
        self_probing_output_log['confidence'] = self_probing_output_log['log_confidence']
        del self_probing_output_log['log_confidence']
        all_outputs_log.append(self_probing_output_log)
    print(f'scene: {','.join(data["environment"])}\ntask: {data["instruction"]}\nself_probing_output: {self_probing_output}\n')
    print('-'*100)
    data['self_probing_output'] = all_outputs
    data['self_probing_output_log'] = all_outputs_log


data = task_dataset[128]
collect_self_prob_outputs(data, force_rerun=True, num_ensemble=1)
print(data['self_probing_output'])
print(data['self_probing_output_log'])





# %%
multi_step_prompt = '''
What will you do? only interact with the objects in the scene.
Read the question, break down the problem into K steps, think step by step, 
give your confidence in each step, and then derive your final answer and your confidence in this answer. 
Note: The confidence indicates how likely you think your answer is true. 
Use the following format to answer: 
"Step 1: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]% 
... 
Step K: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]% 
"
'''
def prompt_multi_step(data, sample_num=1, sample_mode='greedy'):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=','.join(data["environment"]))},
        {"role": "user", "content": "{task}".format(task=data["instruction"])},
        {"role": "user", "content": multi_step_prompt},
    ]
    print(messages)
    response, messages = chat(messages, max_length=1000)
    messages_for_confidence = copy.deepcopy(messages)
    response_for_confidence = response
    print(response)
    content = '''according to your response, what action will you do? reply with the following format:
"Pick object:[the object you will pick(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]
Relation:[the relation between the object you will pick and the object you will put it(should be one of 'in' 'front', 'back', 'left', 'right')]
Target_obj:[the object you will put it(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]"
    '''
    messages.append({
        'role': 'user',
        'content': content
    })
    response, messages = chat(messages, max_length=1000)
    print('*'*100)
    print(response)
    for line in response.split('\n'):
        if 'Pick object:' in line:
            pick_obj = line.split('Pick object:')[1].strip()
        if 'Relation:' in line:
            relation = line.split('Relation:')[1].strip()
        if 'Target_obj:' in line:
            target_obj = line.split('Target_obj:')[1].strip()
    
    confidence = 1
    ress = response_for_confidence.split('Step ')
    for res in ress:
        if not '%' in res:
            continue
        res = res.split('%')[0]
        confidence *= float(res.split(':')[-1])/100
    output_confidence = confidence

    multi_step_output = {
        'pick_obj': pick_obj,
        'relation': relation,
        'target_obj': target_obj,
        'confidence': output_confidence
    }
    return multi_step_output

def collect_multi_step_output(data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'multi_step_output' in data and not force_rerun:
        return
    all_outputs = []
    for i in range(num_ensemble):
        multi_step_output = prompt_multi_step(data)
        all_outputs.append(multi_step_output)
    print(f'scene: {','.join(data["environment"])}\ntask: {data["instruction"]}\nmulti_step_output: {multi_step_output}\n')
    print('-'*100)
    data['multi_step_output'] = all_outputs
    return data['multi_step_output']

data = task_dataset[128]
collect_multi_step_output(data, force_rerun=True, num_ensemble=1)


# %%
top_k_prompt = '''
What will you do? only interact with the objects in the scene.
Provide your k best guesses and the probability that each is correct (0% to 100%) for the following question. 
Give ONLY the task output description of your guesses and probabilities, no other words or explanation. 
Note: The action description should be in the following format:"Pick object:[the object you will pick(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]
Relation:[the relation between the object you will pick and the object you will put it(should be one of 'in' 'front', 'back', 'left', 'right')]
Target_obj:[the object you will put it(should be one of 'green block', 'blue block', 'yellow block', 'green bowl', 'blue bowl', 'yellow bowl')]
"
Your answer should be in the following format: 
"G1: <ONLY the action description of first most likely action> 
P1: <ONLY the probability that G1 is correct, please only include the numerical number in the range of 0-100> 
... 
Gk: <ONLY the action description of k-th most likely action> 
Pk: <ONLY the probability that Gk is correct, please only include the numerical number in the range of 0-100">"
'''
def prompt_top_k(data, sample_num=1, sample_mode='greedy'):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=','.join(data["environment"]))},
        {"role": "user", "content": "{task}".format(task=data["instruction"])},
        {"role": "user", "content": top_k_prompt},
    ]
    response, messages = chat(messages, max_length=1000)
    print(response)
    '''
    G1: Bring the Coke
    P1: 60%

    G2: Bring the Sprite
    P2: 20%

    G3: Bring the bottled unsweetened tea
    P3: 20%
    '''
    # split the response into G1, P1, G2, P2, ...
    top_k_result = {}
    response = response.split('\n')
    for i in range(len(response)):
        if response[i].startswith('G') or response[i].startswith('P'):
            index = int(response[i].split(':')[0][1:])
            if index not in top_k_result:
                top_k_result[index] = {}

        if response[i].startswith('P'):
            confidence = float(response[i].split(':')[-1].strip().split('%')[0])/100
            top_k_result[index]['confidence'] = confidence
        elif response[i].startswith('G'):#G5: Pick object:yellow block Relation:in Target_obj:green bowl
            pick_obj = response[i].split('Pick object:')[1].split('Relation:')[0].strip()
            relation = response[i].split('Relation:')[1].split('Target_obj:')[0].strip()
            target_obj = response[i].split('Target_obj:')[1].strip()
            top_k_result[index]['pick_obj'] = pick_obj
            top_k_result[index]['relation'] = relation
            top_k_result[index]['target_obj'] = target_obj

    output_v = ''
    output_confidence = -9999
    for k, v in top_k_result.items():
        if v['confidence'] > output_confidence:
            output_v = v
            output_confidence = v['confidence']

    cot_output = {
        'pick_obj': output_v['pick_obj'],
        'relation': output_v['relation'],
        'target_obj': output_v['target_obj'],
        'confidence': output_confidence,
        'all_result': top_k_result
    }
    return cot_output


def collect_top_k_outputs(data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'top_k_output' in data and not force_rerun:
        return
    all_outputs = []
    for i in range(num_ensemble):
        top_k_output = prompt_top_k(data)
        all_outputs.append(top_k_output)
    print(f'scene: {','.join(data["environment"])}\ntask: {data["instruction"]}\ntop_k_output: {top_k_output}\n')
    print('-'*100)
    data['top_k_output'] = all_outputs

data = task_dataset[128]
collect_top_k_outputs(data, force_rerun=True, num_ensemble=1)


# %%
import random
def process_mc_raw(mc_raw, add_mc='an option not listed here'):
    mc_all = mc_raw.split('\n')
    mc_processed_all = []
    for mc in mc_all:
        mc = mc.strip()  # sometimes there is leading space
    # skip nonsense
        if len(mc) < 5 or mc[0] not in [
        'a', 'b', 'c', 'd', 'A', 'B', 'C', 'D', '1', '2', '3', '4'
    ]:
            continue
        mc = mc[2:]  # remove a), b), ...
        mc = mc.strip().lower().split('.')[0]
        mc_processed_all.append(mc)
    if len(mc_processed_all) < 4:
        raise 'Cannot extract four options from the raw output.'
# Check if any repeated option - use do nothing as substitue
    mc_processed_all = list(set(mc_processed_all))
    if len(mc_processed_all) < 4:
        num_need = 4 - len(mc_processed_all)
        for _ in range(num_need):
            mc_processed_all.append('do nothing')
    prefix_all = ['A) ', 'B) ', 'C) ', 'D) ']
    if add_mc is not None:
        mc_processed_all.append(add_mc)
        prefix_all.append('E) ')
    random.shuffle(mc_processed_all)
    mc_prompt = ''
    for mc_ind, (prefix, mc) in enumerate(zip(prefix_all, mc_processed_all)):
        mc_prompt += prefix + mc
        if mc_ind < len(mc_processed_all) - 1:
            mc_prompt += '\n'
    add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
    return mc_prompt, mc_processed_all, add_mc_prefix

def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

# apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx

demo_mc_gen_prompt0 = """
We: You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.

We: On the table there are these objects: blue block, yellow bowl, yellow block, green bowl, green block, blue bowl.
We: Now, put the yellow bowl close to the blue blue block
You: These are some options:
A) Pick object:yellow bowl Relation:front Target_obj:blue block
B) Pick object:yellow bowl Relation:back Target_obj:blue block
C) Pick object:yellow bowl Relation:left Target_obj:blue block
D) Pick object:yellow bowl Relation:right Target_obj:blue block

We: On the table there are these objects: yellow bowl, green bowl, green block, yellow block, blue block, blue bowl.
We: Now, put the yellow bowl ot the front of green block
You: These are some options:
A) Pick object:yellow block Relation:front Target_obj:green block
B) Pick object:yellow block Relation:back Target_obj:green block
C) Pick object:yellow block Relation:left Target_obj:green block
D) Pick object:green block Relation:right Target_obj:yellow block

We: On the table there are these objects: blue bowl, yellow block, green bowl, blue block, green block, yellow bowl.
We: Now, put the yellow bowl to the left of the block
You: These are some options:
A) Pick object:yellow bowl Relation:front Target_obj:green block
B) Pick object:yellow bowl Relation:left Target_obj:yellow block
C) Pick object:yellow bowl Relation:left Target_obj:blue block
D) Pick object:yellow bowl Relation:right Target_obj:green block

We: On the table there are these objects: blue bowl, yellow block, green bowl, blue block, green block, yellow bowl.
We: Now, put the yellow block in the yellow bowl
You: These are some options:
A) Pick object:yellow block Relation:in Target_obj:yellow bowl
B) Pick object:yellow block Relation:left Target_obj:green bowl
C) Pick object:yellow block Relation:left Target_obj:blue bowl
D) Pick object:blue block Relation:in Target_obj:green bowl

We: On the table there are these objects: green bowl, yellow block, blue bowl, yellow bowl, green block, blue block.
We: Now, {instruction}
You: These are some options:
"""


# conformal prediction 方法输出不确定性
def generate_multiple_choice(data):
    instruction = data['instruction']
    # scene_objects = data['scene_objects']
    result = {}
    # skip_calibration = False #@param {type:"boolean"}
    # if skip_calibration: qhat = 0.928 # based on epsilon=0.2

    demo_mc_gen_prompt = demo_mc_gen_prompt0
# prompt for generating multiple choice
    demo_mc_gen_prompt = demo_mc_gen_prompt.replace('{instruction}', instruction)
    # demo_mc_gen_prompt = demo_mc_gen_prompt.replace('{scene_objects}', scene_objects)

# Generate multiple choices
    _, demo_mc_gen_raw = llama(demo_mc_gen_prompt, stop_seq=['\n\n', 'We:'])
    demo_mc_gen_raw = demo_mc_gen_raw.strip()
    demo_mc_gen_full, demo_mc_gen_all, demo_add_mc_prefix = process_mc_raw(demo_mc_gen_raw)

# get the part of the current scenario from the previous prompt
    demo_cur_scenario_prompt = demo_mc_gen_prompt.split('\n\n')[-1].strip()

# get new prompt
    demo_mc_score_background_prompt = """
    We: You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.

We: On the table there are these objects: green bowl, yellow block, blue bowl, yellow bowl, green block, blue block.
We: Now, {instruction}
You: These are some options:
{mc}
We: Which option is correct? Answer with a single letter.
You:
    """.strip().replace('{instruction}', instruction).replace('{mc}', demo_mc_gen_full)
    # demo_mc_score_prompt = demo_mc_score_background_prompt + '\n\n' + demo_cur_scenario_prompt + '\n' + demo_mc_gen_full
    # demo_mc_score_prompt += "\nWe: Which option is correct? Answer with a single capital letter."
    # demo_mc_score_prompt += "\nYou:"
    demo_mc_score_prompt = demo_mc_score_background_prompt.strip()
    print(demo_mc_score_prompt)

# scoring
# mc_score_response, _ = lm(demo_mc_score_prompt, max_tokens=1, logprobs=5)
# top_logprobs_full = mc_score_response["choices"][0]["logprobs"]["top_logprobs"][0]
# top_tokens = [token.strip() for token in top_logprobs_full.keys()]
# top_logprobs = [value for value in top_logprobs_full.values()]


# scoring with llama-----------------------------------------------------
    mc_score_response, response = llama(demo_mc_score_prompt, max_length=1, output_scores=True, processors=processors)
    print(response)

# Get the logits of the last token generated
    last_token_logits = mc_score_response.scores[-1]
    last_token_logits = last_token_logits.detach().cpu()

# Apply softmax to convert logits to probabilities
    probs = torch.softmax(last_token_logits, dim=-1)
    log_probs = torch.log(probs)

# Extract probabilities for 'A', 'B', 'C'
    all_tokens = ['A', 'B', 'C', 'D', 'E']
    allowed_token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    token_probs = []
    for i in range(len(all_tokens)):
        log_prob = log_probs[0, allowed_token_ids[i]].item()
        token_probs.append((all_tokens[i], log_prob))

    
    

# Collect and sort probabilities
    sorted_token_probs = sorted(token_probs, key=lambda x: x[1], reverse=True)
    top_tokens = [tuple[0] for tuple in sorted_token_probs]
    top_logprobs = [tuple[1] for tuple in sorted_token_probs]
    for jj in range(len(top_tokens)):
        print(top_tokens[jj], top_logprobs[jj])
    
# scoring with llama end ---------------------------------------------------
# get prediction set

    mc_smx_all = temperature_scaling(top_logprobs, temperature=5)

# include all options with score >= 1-qhat
    prediction_set = [
          token for token_ind, token in enumerate(top_tokens)
        #   if mc_smx_all[token_ind] >= 1 - qhat
      ]

# print
    print('Multiple choices generated:')
    print(demo_mc_gen_full)
    print('\nPrediction set:', prediction_set)
    print('token', 'prob')
    all_result = {}
    token_mc_dict = {}
    for mc in demo_mc_gen_full.split('\n'):
        if len(mc) < 1:
            continue
        if mc[0] in ['A', 'B', 'C', 'D', 'E']:
            token_mc_dict[mc[0]] = mc[2:].strip()
    for token_ind, token in enumerate(top_tokens):
        print(token, mc_smx_all[token_ind])
        action = token_mc_dict[token]
        if 'pick object' not in action:
            pick_obj = relation = target_obj = ''
        else:
            pick_obj = action.split('pick object:')[1].split('relation:')[0].strip()
            relation = action.split('relation:')[1].split('target_obj:')[0].strip()
            target_obj = action.split('target_obj:')[1].strip()
        all_result[token_ind+1]={
            'pick_obj': pick_obj,
            'relation': relation,
            'target_obj': target_obj,
            'confidence': float(mc_smx_all[token_ind])
        }
    
    result['all_result'] = all_result

    result['pick_obj'] = all_result[1]['pick_obj']
    result['relation'] = all_result[1]['relation']
    result['target_obj'] = all_result[1]['target_obj']
    result['confidence'] = mc_smx_all[0]
    return result

def collect_conformal_outputs(data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'conformal_output' in data and not force_rerun:
        return
    all_outputs = []
    for i in range(num_ensemble):
        conformal_prediction_output = generate_multiple_choice(data)
        all_outputs.append(conformal_prediction_output)
    print(f'scene: {','.join(data["environment"])}\ntask: {data["instruction"]}\nconformal_prediction_output: {conformal_prediction_output}\n')
    print('-'*100)
    data['conformal_output'] = all_outputs
    return data['conformal_output']
collect_conformal_outputs(task_dataset[2], num_ensemble=1, force_rerun=True)

# %%

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

evaluate_output_confidence1(task_dataset[0])

# %%
def all_process_method(data):
    # 收集不同测试方法数据
    print('collect_vanilla_outputs'+'-'*100)
    collect_vanilla_outputs(prompt_vanilla, data)
    print('collect_cot_output'+'-'*100)
    collect_cot_output(data)
    print('collect_self_prob_outputs'+'-'*100)
    collect_self_prob_outputs(data)
    print('collect_multi_step_output'+'-'*100)
    collect_multi_step_output(data)
    print('collect_top_k_outputs'+'-'*100)
    collect_top_k_outputs(data)
    print('collect_conformal_outputs'+'-'*100)
    collect_conformal_outputs(data, force_rerun=True)
    print('align_atomic_actions'+'-'*100)
    evaluate_output_confidence1(data)


# %%
data = task_dataset[1]
all_process_method(data)
import json
print(json.dumps(data['result'], indent=1))

# %%

# task_dataset = task_dataset[:1]
fail_list = []
from tqdm.notebook import tqdm
for data_index in tqdm(range(len(task_dataset))):
    data = task_dataset[data_index]
    pickle_path = f"./cache/tablet/data{data['index']}.pkl"
    if os.path.exists(pickle_path):
        # # --------- 为了重新运行其中一些内容 还有force_rerun参数记得改
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        task_dataset[data_index] = data
        # # ---------- 结束
        continue
    # error_path = f'./cache/error{data["index"]}.pkl'
    # if os.path.exists(error_path):
    #     continue
    print('-'*20, f'开始处理新情景[{data["index"]}]', '-'*20)
    try:
        all_process_method(data)
        # save data to ./cache/data[index].pkl
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        # raise e
        with open(f'./cache/error{data["index"]}.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(e)
        fail_list.append(data['index'])
        continue
print(fail_list)



# %%
# 下一步去cache_check文件运行


