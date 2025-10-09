# %%
import os
TEST_FLAG = os.getenv('TEST_FLAG', '0') == '1'
cache_dir = os.getenv('CACHE_DIR', '0')
if cache_dir == "0":
    cache_dir = None

# %%
#@markdown A few imports and downloading data
# !pip install -U --no-cache-dir gdown --pre
# !pip install openai tqdm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# import openai
import signal
import tqdm.notebook as tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import pdb
import pickle

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
class LLAMA:
    def __init__(self, model_name,load_in_8bit=False):
        self.device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          low_cpu_mem_usage=True, 
                                                          torch_dtype=torch.float16, 
                                                          device_map="auto",
                                                          cache_dir=cache_dir,
                                                          load_in_8bit=load_in_8bit)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

    def llama(self, prompt, max_length=256, output_scores=False, processors=None, temperature=1.0, stop_seq=None, skip_inputs=True):
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            # 停止序列处理
        generation_kwargs = {}
        if stop_seq:
            stop_token_ids = self.tokenizer(stop_seq, add_special_tokens=False).input_ids

            # Define stopping criteria
            from transformers import StoppingCriteria, StoppingCriteriaList

            class StopOnTokenCriteria(StoppingCriteria):
                def __init__(self, stop_sequences):
                    # stop_sequence should be a list of token IDs representing \N\N
                    self.stop_sequences = stop_sequences

                def __call__(self, input_ids, scores, **kwargs):
                    # Check if the end of input_ids matches the stop_sequence
                    for stop_sequence in self.stop_sequences:  
                        if len(input_ids[0]) >= len(stop_sequence):  # Ensure there are enough tokens to compare
                            if input_ids[0, -len(stop_sequence):].tolist() == stop_sequence:
                                return True
                    return False

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokenCriteria(stop_token_ids)])
        outputs = self.model.generate(**inputs, logits_processor=processors, 
                                max_length=inputs.input_ids.size(1) + max_length, 
                                return_dict_in_generate=True, 
                                output_scores=output_scores, 
                                temperature=temperature, 
                                pad_token_id=self.tokenizer.eos_token_id,
                                do_sample=False,
                                **generation_kwargs)
        
        if skip_inputs:# 将output中的input删除，只保留新生成的output
            new_generate_sequence = outputs.sequences[0, inputs.input_ids.size(1):]
            decoded_output = self.tokenizer.decode(new_generate_sequence)
        else:
            decoded_output = self.tokenizer.decode(outputs.sequences[0])
        return outputs, decoded_output

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor
from transformers import LogitsProcessorList, LogitsProcessor
import torch

try:
    llama_obj
except:
    model_name = "unsloth/Llama-3.3-70B-Instruct"
    if TEST_FLAG:
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
print(chat(messages))

# %%

scenario_info_path = 'content/metabot-tasks-info.txt'
with open(scenario_info_path, 'r') as f:
    scenario_info_text = f.read()
scenario_info_text = scenario_info_text.split('\n\n')
print('Loaded scenario info from ' + scenario_info_path)
#@title
# Print a scenario
print('Sample scenario:\n')
print(scenario_info_text[0].split('\n',1)[1])  # remove the printed index
'''
Scene: a Coke, a bottled unsweetened tea, and a Sprite
Task: Bring me a flavored drink.
User intent (object): Coke, bottled unsweetened tea, Sprite
User intent (location): pick-up
Scene objects: Coke, bottled unsweetened tea, Sprite
Task category: creative_multilabel_task'''

scenario_data = []
for scenario in scenario_info_text:
    if len(scenario.split('\n')) < 7:
        continue
    index = int(scenario.split('\n')[0])
    scenario = scenario.split('\n')[1:]
    scene = scenario[0].split('Scene: ')[1]
    task = scenario[1].split('Task: ')[1]
    user_intent_object = scenario[2].split('User intent (object): ')[1]
    user_intent_location = scenario[3].split('User intent (location): ')[1]
    scene_objects = scenario[4].split('Scene objects: ')[1]
    task_category = scenario[5].split('Task category: ')[1]
    scenario_data.append({
            'index': index,
            'scene': scene,
            'task': task,
            'user_intent_object': user_intent_object,
            'user_intent_location': user_intent_location,
            'scene_objects': scene_objects,
            'task_category': task_category
    })

print(len(scenario_data))
print(scenario_data[0])

# %%
# run local
import requests
import json
import hashlib
import os
prompt = '''
You are a robot in kitchen. You are given a scenario and a task. You need to place an item in a new location.
Scenario: {scene}
Task: {task}
Optional items are:  {scene}
Optional target locations are:  {scene}, user`s hand, top drawer, bottom drawer, garbage can.
Please tell me what items you choose from and where is the target locations. You can choose one or more. You need to choose every item and target location fit the task.
(Based on common-sense reasoning, extreme special cases should be disregarded.) '''

def ambigous_check(scenario):
    scence = scenario['scene']
    task = scenario['task']
    input_prompt = prompt.format(scene=scence, task=task)
    print(input_prompt)
    messages = [
        {
            "role": "user",
            "content": input_prompt
        }
    ]

    res = chat(messages, do_sample=False)[0]
    print(res)
    new_p = f'''
    The following is a robot's thought process; how many items he choose? and how many locations he choose? Please answer with only two numbers, separated by a space, with the item number first.
    {res}
    '''
    messages = [
        {
            "role": "user",
            "content": new_p
        }
    ]
    res2 = chat(messages, do_sample=False)[0]
    print(res2)
    nums = res2.split(' ')
    amb = False
    if int(nums[0]) == 1 and int(nums[1]) == 1:
        amb = False
    else:
        amb = True
    result = {
        'ambiguous': amb
    }
    return result



ambigous_check(scenario_data[0])

# %%
import tqdm.notebook as tqdm
fail_list = []
task_len = 300
if TEST_FLAG: task_len = 5
for i in tqdm.trange(task_len):
    file_path = f'content/task_data_llama/{i}.json'
    if os.path.exists(file_path):
        continue
    res = ambigous_check(scenario_data[i])
    if res is None:
        fail_list.append(i)
        continue
    scenario_data[i].update(res)
    with open(file_path, 'w') as f:
        json.dump(scenario_data[i], f, indent=1)
print(fail_list)

# %%



