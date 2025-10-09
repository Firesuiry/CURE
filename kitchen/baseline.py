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
# dataset files - small sizes
# !gdown https://drive.google.com/uc?id=1XWIeGfF08V1eR104VLDilmwhIGVk2uzk
# !gdown https://drive.google.com/uc?id=1iEIZaVbbajMXsNdrjVkOgK5rhPtfl5WI

# Set OpenAI API key.
# openai.api_key = openai_api_key

# %%
NUM_ENSEMBLE= 1
if TEST_FLAG:
    NUM_ENSEMBLE = 1
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
    model_name = "unsloth/Llama-3.3-70B-Instruct"
    if TEST_FLAG:
        model_name = "voidful/Llama-3.2-8B-Instruct"
    llama_obj = LLAMA(model_name, load_in_8bit=True, cache_dir=cache_dir)
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
#@title
# Load the 300 scenarios
run_flag = True
try:
    scenario_data
    run_flag = False
except NameError:
  pass

if run_flag:
  scenario_info_path = './content/metabot-tasks-info.txt'
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
    # if debug:
    #   if len(scenario_data) > 1:
    #     break
  print(len(scenario_data))
  print(scenario_data[0])

# %%
all_scence = '''You are a robot operating in an office kitchen. 
You are in front of a counter with two closed drawers, a top one and a bottom one. 
There is also a landfill bin, a recycling bin, and a compost bin.
'''.replace('\n', '')

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
Note: The action you take should be one of [pick-up, bring, move, open, close] and something you seen(should be a single full name) [to where(should be a single full name) you seen if needed].
Use the following format to answer: 
"Action:[What will you do]
Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%"
Only the answer and confidence, don't give me the explanation.  
Now, please answer this question and provide your confidence level.
'''
def prompt_vanilla(data, sample_num=1, sample_mode='greedy', print_flag = False):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=data["scene"])},
        {"role": "user", "content": "{task}".format(task=data["task"])},
        {"role": "user", "content": vanilla_prompt},
    ]
    response, messages = chat(messages, max_length=1000)

    if print_flag:print(response)
    for line in response.split('\n'):
        if 'Action:' in line:
            output_action = line.split('Action:')[1].strip()
        if 'Confidence:' in line:
            output_confidence = float(line.split('Confidence:')[1].strip().replace('%', ''))/100
    
    vanilla_output = {
        'action': output_action,
        'confidence': output_confidence
    }
    return vanilla_output
def collect_vanilla_outputs(prompt_vanilla, data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'vanilla_output' in data and not force_rerun:
        return
    print(f'scene: {data["scene"]}\ntask: {data["task"]}\n')
    all_outputs = []
    for i in range(num_ensemble):
        vanilla_output = prompt_vanilla(data)
        all_outputs.append(vanilla_output)
    print(f'scene: {data["scene"]}\ntask: {data["task"]}\nvanilla_output: {vanilla_output}\n')
    print('-'*100)
    data['vanilla_output'] = all_outputs


# data = scenario_data[128]
# collect_vanilla_outputs(prompt_vanilla, data, force_rerun=True, num_ensemble=1)


# %%
# COT 方法输出不确定性
cot_prompt = '''
analyze step by step, provide your what will you do and your confidence in this answer. 
'''
cot_prompt = '''
analyze step by step, provide what will you do and your confidence in this answer. 
Note: The confidence indicates how likely you think your action is true. Use the following format to answer: 
"Explanation: [insert step-by-step analysis here] 
Action:[What will you do Here]
Confidence:[Your confidence level, please only include the numerical number in the range of 0-100]%"
Note: The action you take should be one of [pick-up, bring, move, open, close] and something you seen(should be a single full name) [to where(should be a single full name) you seen if needed].
Only give me the reply according to this format, don't give me any other words. 
Now, please answer this question and provide your confidence level. Let's think it step by step.
'''
def prompt_cot(data, sample_num=1, sample_mode='greedy'):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=data["scene"])},
        {"role": "user", "content": "{task}".format(task=data["task"])},
        {"role": "user", "content": cot_prompt},
    ]
    response, messages = chat(messages, max_length=1000)
    messages_for_confidence = copy.deepcopy(messages)
    print(response)
    for line in response.split('\n'):
        if 'Action:' in line:
            output_action = line.split('Action:')[1].strip()
        if 'Confidence:' in line:
            output_confidence = float(line.split('Confidence:')[1].strip().replace('%', ''))/100
    
    cot_output = {
        'action': output_action,
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
    print(f'scene: {data["scene"]}\ntask: {data["task"]}\ncot_output: {cot_output}\n')
    print('-'*100)
    data['cot_output'] = all_outputs

# data = scenario_data[128]
# collect_cot_output(data, force_rerun=True, num_ensemble=1)

# %%
self_probing_prompt = '''
What will you do? 
Note: The action you take should be one of [pick-up, bring, move, open, close] and something you seen(should be a single full name) [to where(should be a single full name) you seen if needed].
'''
def prompt_self_probing(data, sample_num=1, sample_mode='greedy'):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=data["scene"])},
        {"role": "user", "content": "{task}".format(task=data["task"])},
        {"role": "user", "content": self_probing_prompt},
    ]
    print(messages)
    response, messages = chat(messages, max_length=1000)
    messages_for_confidence = copy.deepcopy(messages)
    messages_for_confidence2 = copy.deepcopy(messages)
    print(response)
    output_action = response
    messages_for_confidence.append({
        'role': 'user',
        'content': 'according to your response, what is your confidence in this action? reply with the confidence only. the confidence should be a percentage.'
    })
    response, messages_for_confidence = chat(messages_for_confidence, max_length=1000)
    print('*'*100)
    print(response)
    output_confidence = float(response.split('%')[0])/100
    cot_output = {
        'action': output_action,
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
    print(f'scene: {data["scene"]}\ntask: {data["task"]}\nself_probing_output: {self_probing_output}\n')
    print('-'*100)
    data['self_probing_output'] = all_outputs
    data['self_probing_output_log'] = all_outputs_log


# data = scenario_data[0]
# collect_self_prob_outputs(data, force_rerun=True, num_ensemble=1)
# print(data['self_probing_output'])
# print(data['self_probing_output_log'])





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
Final Answer and Overall Confidence (0-100): [ONLY the answer type; not a complete sentence], [Your confidence value]%"
'''
def prompt_multi_step(data, sample_num=1, sample_mode='greedy'):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=data["scene"])},
        {"role": "user", "content": "{task}".format(task=data["task"])},
        {"role": "user", "content": multi_step_prompt},
    ]
    print(messages)
    response, messages = chat(messages, max_length=1000)
    messages_for_confidence = copy.deepcopy(messages)
    response_for_confidence = response
    print(response)
    messages.append({
        'role': 'user',
        'content': 'according to your response, what action will you do? reply with the action only.Note: The action you take should be one of [pick-up, bring, move, open, close] and something you seen(should be a single full name) [to where(should be a single full name) you seen if needed].'
    })
    response, messages = chat(messages, max_length=1000)
    print('*'*100)
    print(response)
    output_action = response
    
    confidence = 1
    ress = response_for_confidence.split('Step ')
    for res in ress:
        if not '%' in res:
            continue
        res = res.split('%')[0]
        confidence *= float(res.split(':')[-1])/100
    output_confidence = confidence

    cot_output = {
        'action': output_action,
        'confidence': output_confidence
    }
    return cot_output

def collect_multi_step_output(data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'multi_step_output' in data and not force_rerun:
        return
    all_outputs = []
    for i in range(num_ensemble):
        multi_step_output = prompt_multi_step(data)
        all_outputs.append(multi_step_output)
    print(f'scene: {data["scene"]}\ntask: {data["task"]}\nmulti_step_output: {multi_step_output}\n')
    print('-'*100)
    data['multi_step_output'] = all_outputs

# data = scenario_data[128]
# collect_multi_step_output(data, force_rerun=True, num_ensemble=1)


# %%
top_k_prompt = '''
What will you do? only interact with the objects in the scene.
Provide your k best guesses and the probability that each is correct (0% to 100%) for the following question. 
Give ONLY the task output description of your guesses and probabilities, no other words or explanation. 
Note: The action you take should be one of [pick-up, bring, move, open, close] and something you seen(should be a single full name) [to where(should be a single full name) you seen if needed].
For example: 
G1: <ONLY the action description of first most likely guess; not a complete sentence, just the guess!> 
P1: <ONLY the probability that G1 is correct, without any extra commentary whatsoever; just the probability!> 
... 
Gk: <ONLY the action description of k-th most likely guess> 
Pk: <ONLY the probability that Gk is correct, without any extra commentary whatsoever; just the probability!>
'''
def prompt_top_k(data, sample_num=1, sample_mode='greedy'):
    messages = [
        {"role": "system", "content": all_scence},
        {"role": "user", "content": "On the counter, there is {scene}".format(scene=data["scene"])},
        {"role": "user", "content": "{task}".format(task=data["task"])},
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
        elif response[i].startswith('G'):
            top_k_result[index]['action']= response[i].split(':')[-1].strip()

    output_action = ''
    output_confidence = -9999
    for k, v in top_k_result.items():
        if v['confidence'] > output_confidence:
            output_action = v['action']
            output_confidence = v['confidence']

    cot_output = {
        'action': output_action,
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
    print(f'scene: {data["scene"]}\ntask: {data["task"]}\ntop_k_output: {top_k_output}\n')
    print('-'*100)
    data['top_k_output'] = all_outputs

# data = scenario_data[128]
# collect_top_k_outputs(data, force_rerun=True, num_ensemble=1)


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
We: You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
We: On the counter, there is an orange soda, a Pepsi, and an apple.
We: Put that drink in the top drawer.
You:
A) open the top drawer and put the orange soda in it
B) open the bottom drawer and put the Pepsi in it
C) open the bottom drawer and put the orange soda in it
D) open the top drawer and put the Pepsi in it
We: On the counter, there is an energy bar, a banana, and a microwave.
We: Put the snack next to the microwave.
You:
A) pick up the energy bar and put it next to the microwave
B) pick up the banana and put it next to the energy bar
C) pick up the banana and put it next to the microwave
D) pick up the energy bar and put it next to the banana
We: On the counter, there is a Coke, a Sprite, and a sponge.
We: Can you dispose of the can? It should have expired.
You:
A) pick up the sponge and put it in the landfill bin
B) pick up the Coke and put it in the recycling bin
C) pick up the Sprite and put it in the recycling bin
D) pick up the Coke and put it in the landfill bin
We: On the counter, there is a bottled water, a bag of jalapeno chips, and a bag of rice chips.
We: I would like a bag of chips.
You:
A) pick up the bottled water
B) pick up the jalapeno chips
C) pick up the kettle chips
D) pick up the rice chips
We: On the counter, there is {scene_objects}
We: {task}
You:
"""


# conformal prediction 方法输出不确定性
def generate_multiple_choice(data):
    instruction = data['task']
    scene_objects = data['scene_objects']
    result = {}
    # skip_calibration = False #@param {type:"boolean"}
    # if skip_calibration: qhat = 0.928 # based on epsilon=0.2

    demo_mc_gen_prompt = demo_mc_gen_prompt0
# prompt for generating multiple choice
    demo_mc_gen_prompt = demo_mc_gen_prompt.replace('{task}', instruction)
    demo_mc_gen_prompt = demo_mc_gen_prompt.replace('{scene_objects}', scene_objects)

# Generate multiple choices
    _, demo_mc_gen_raw = llama(demo_mc_gen_prompt, stop_seq=['\n\n', 'We:'])
    demo_mc_gen_raw = demo_mc_gen_raw.strip()
    demo_mc_gen_full, demo_mc_gen_all, demo_add_mc_prefix = process_mc_raw(demo_mc_gen_raw)

# get the part of the current scenario from the previous prompt
    demo_cur_scenario_prompt = demo_mc_gen_prompt.split('\n\n')[-1].strip()

# get new prompt
    demo_mc_score_background_prompt = """
    You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a middle one. There is also a landfill bin, a recycling bin, and a compost bin.
    """.strip()
    demo_mc_score_prompt = demo_mc_score_background_prompt + '\n\n' + demo_cur_scenario_prompt + '\n' + demo_mc_gen_full
    demo_mc_score_prompt += "\nWe: Which option is correct? Answer with a single capital letter."
    demo_mc_score_prompt += "\nYou:"

# scoring
# mc_score_response, _ = lm(demo_mc_score_prompt, max_tokens=1, logprobs=5)
# top_logprobs_full = mc_score_response["choices"][0]["logprobs"]["top_logprobs"][0]
# top_tokens = [token.strip() for token in top_logprobs_full.keys()]
# top_logprobs = [value for value in top_logprobs_full.values()]


# scoring with llama-----------------------------------------------------
    mc_score_response, response = llama(demo_mc_score_prompt, max_length=1, output_scores=True, processors=processors)

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
        all_result[token_ind+1]={
            'action': token_mc_dict[token],
            'confidence': mc_smx_all[token_ind]
        }
    
    result['all_result'] = all_result
    result['action'] = all_result[1]['action']
    result['confidence'] = mc_smx_all[0]
    return result

def collect_conformal_outputs(data, force_rerun=False, num_ensemble=NUM_ENSEMBLE):
    if 'conformal_output' in data and not force_rerun:
        return
    all_outputs = []
    for i in range(num_ensemble):
        conformal_prediction_output = generate_multiple_choice(data)
        all_outputs.append(conformal_prediction_output)
    print(f'scene: {data["scene"]}\ntask: {data["task"]}\nconformal_prediction_output: {conformal_prediction_output}\n')
    print('-'*100)
    data['conformal_output'] = all_outputs

# print(generate_conformal_prediction(scenario_data[128], num_ensemble=1, force_rerun=True))

# %%
# 动作对齐
action_example = '''
"action1:'pick up the coke', action2:'pick up the sprite',No
action1:'Bring the Coke', action2:'I will bring you the Coke',Yes
action1:'Retrieve the Sprite from the counter', action2:'Bring you the Sprite',Yes
action1:'Bring the Coke', action2:'Pick the Coke',Yes
action1:'Bring the Coke', action2:'I will retrieve the Coke from the counter and hand it to you.',Yes
action1:'Bring you the Sprite', action2: 'Bring you a Coke',No"
'''
instruction_prompt = '''
Similar actions are actions that have the object and simliar action. 
For example, 
"pick up the coke" and "bring the coke" are similar actions. 
"bring orange to bottled water" and "move orange to bottled water" are similar actions.
"bring the coke" and "bring the sprite" are not similar actions. 

If two actions want to move the object to a location and the locations are same.
Then the two actions are similar.
For example,
"bring the coke to the table" and "pick up the coke to the table" are similar actions.
"bring the coke to the landfill bin" and "pick up the coke to the recycling bin" are not similar actions.
"bring the coke to the landfill bin" and "pick up the coke" are not similar actions.

If two actions want to move same object to you or your hand, then the two actions are similar.
For example,
"bring the coke to you" and "pick up the coke to your hand" are similar actions.
"bring the coke to you" and "pick up the coke to here" are similar actions.
"bring the coke to you" and "pick up the sprite to here" are not similar actions.
'''

def check_if_action_same(action1, action2):
    def format(action):
        action = action.lower()
        action = action.replace('the', '')
        action = action.replace('a', '')
        action = action.replace('an', '')
        action = action.replace('to you', '')
        action = action.replace('to here', '')
        action = action.replace('to user', '')
        action = action.replace('bring', 'pick up')
        action = action.replace('pick-up', 'pick up')
        action = action.replace('move', 'pick up')
        action = action.replace('put', 'pick up')
        action = action.replace('I will ', '')
        delete_words = ['bag of', 'box of', 'bottle of']
        for word in delete_words:
            action = action.replace(word, '')
        while '  ' in action:
            action = action.replace('  ', ' ')
        action = action.strip()
        return action
    return format(action1) == format(action2)

def simple_action_classification(actions):
    small_actions = actions.copy()
    small_classes = []
    while len(small_actions) > 0:
        action2 = small_actions.pop(0)
        small_class = [action2]
        for i in range(len(small_actions)):
            if check_if_action_same(action2, small_actions[i]):
                small_class.append(small_actions[i])
        for ac in small_class[1:]:
            small_actions.remove(ac)
        small_classes.append(small_class)
    return small_classes


def classify_similar_actions(all_outputs):
    actions = list(all_outputs)
    # print(actions)
    action_classes = []
    while len(actions) > 0:
        action1 = actions.pop(0)
        print(f'l{len(actions)}',end=' ')
        action_class = [action1]
        # 提前采用简单方法进行分类
        small_classes = simple_action_classification(actions)

        for small_class in small_classes:
            act = small_class[0]
            if check_if_action_same(action1, act):
                response = "yes"
            else:
                messages = [
                    {"role": "system", "content": "you are a artificial intelligence assistant, you need to answer the following question"},
                    {"role": "user", "content": f"you need to answer if the following two action descriptions are similar. {instruction_prompt}"},
                    {"role": "user", "content": "Are the following two action descriptions similar? action1:'{}' action2:'{}'".format(act, action1)},
                    {"role": "user", "content": "reply with Yes or No ONLY. do not give any explanation."},
                ]
                response, messages = chat(messages, max_length=1000, do_sample=False)
                # print(response)
            if 'yes' in response.lower():
                action_class.extend(small_class)
        for action in action_class[1:]:
            actions.remove(action)
        action_classes.append(action_class)


    #     for i in range(len(actions)):
    #         print('|', end='')
    #         if check_if_action_same(actions[i],action1):
    #             response = "yes"
    #         else:
    #             messages = [
    #                 {"role": "system", "content": "you are a artificial intelligence assistant, you need to answer the following question"},
    #                 # {"role": "user", "content": "you need to answer if the following two action descriptions are similar. Some examples are as follows:{}".format(action_example)},
    #                 {"role": "user", "content": f"you need to answer if the following two action descriptions are similar. {instruction_prompt}"},
    #                 {"role": "user", "content": "Are the following two action descriptions similar? action1:'{}' action2:'{}'".format(actions[i], action1)},
    #                 {"role": "user", "content": "reply with Yes or No"},
    #             ]
    #             response, messages = chat(messages, max_length=1000, do_sample=False)
    #         if 'yes' in response.lower():
    #             action_class.append(actions[i])
    #     action_classes.append(action_class)
    #     for action in action_class[1:]:
    #         actions.remove(action)
    # for action_class in action_classes:
    #     print(action_class)
    return action_classes
# a = ['pick-up Pepsi to user', 'pick-up Coke to you']
# classify_similar_actions(a)

# %%
# 动作原子化
atomic_prompt = '''
NOTE:"You need to reply with the following format:
"action:[the action will be done, with a single action, not a sentence, SHOULD be the action in above content]
object:[the object that the action will be done on, with a single object, not a sentence, should be the object in above content, should NOT have verbs in the object]
from location:[the location where the action will be done, with a single location, not a sentence, should be in above content, say NULL if no from location]
to location:[the location where the object will be moved to, with a single location, not a sentence, should be in above content]"
you can reply with "NULL" for the space you are not sure.
You need to reply with the words appeared in the above sentence."
'''
def make_action_atomic(action, scene_objects):
    messages = [
        {"role": "system", "content": "you are a artificial intelligence assistant, you need to answer the following question"},
        {"role": "user", "content": "read the sentence:'{}', what is the action it done?".format(action)},
        {"role": "user", "content": atomic_prompt},
    ]
    response, messages = chat(messages, max_length=1000, do_sample=False)
    # print(response)
    action_data = {
        'action': response.split('action:')[1].split('\n')[0].strip().lower(),
        'object': response.split('object:')[1].split('from location:')[0].strip().lower(),
        'from_location': response.split('from location:')[1].split('to location:')[0].strip().lower(),
        'to_location': response.split('to location:')[1].strip().lower(),
    }
    for obj in scene_objects:
        if obj.lower() in action_data['object']:
            action_data['object'] = obj.strip()
            break

    def location_format(location):
        delete_words = ['next to ', 'next ', 'besides ', 'beside ','the ']
        for word in delete_words:
            location = location.replace(word, '')
        return location
    action_data['from_location'] = location_format(action_data['from_location'])
    action_data['to_location'] = location_format(action_data['to_location'])
    should_not_have_words = ['user', 'you', 'here']
    # if any in action_data['to_location']:
    if any([word in action_data['to_location'] for word in should_not_have_words]):
        action_data['to_location'] = 'null'
    action_data['reconstructed_action'] = action_data['action'] + ' ' + action_data['object']
    if action_data['to_location'] and 'null' not in action_data['to_location'].lower():
        action_data['reconstructed_action'] += ' to ' + action_data['to_location']
    return action_data

# actions = ['Open bottom drawer', 'Dispose of the orange soda in the landfill bin.', 'Open the landfill bin and dispose of the orange soda.', 'Throw orange soda into recycling bin', 'Throw the orange soda away in the landfill bin.', 'Open top drawer', "I will dispose of the orange soda in the landfill bin, considering the contents, and then place the container in the recycling bin if it's recyclable.", 'I will dispose of the orange soda in the recycling bin.', 'Dispose of the orange soda in the recycling bin.', 'I will throw the orange soda into the landfill bin.', 'I will throw the orange soda into the landfill bin', 'Throw orange soda into compost bin', 'Throw orange soda into landfill bin', 'Throw the orange soda in the landfill bin.','Bring the Coke', 'Bring the Coke.', 'I will pick up the Coke.', 'I will bring the Coke.', 'Pick up the Coke', 'Pick up the Coke.', 'Bring the bottled unsweetened tea', 'Bring the Sprite', 'I will bring the Coke or the Sprite', 'Open the top drawer.', 'Pick up the Coke from the counter.']
# atomic_actions = []
# for action in actions:
#     print(action)
#     atomic_action_data = make_action_atomic(action)
#     atomic_actions.append(atomic_action_data)
#     for k,v in atomic_action_data.items():
#         print(f'{k}: {v}')
#     print('*'*100)

# reconstructed_actions = [data['reconstructed_action'] for data in atomic_actions]
# action_classes = classify_similar_actions(reconstructed_actions)
# # 展示分类结果
# for action_class in action_classes:
#     print('-'*100)
#     print(action_class)

# action = 'pick-up rice chips to landfill bin'
# make_action_atomic(action)


# %%
# 动作对齐
def action_reformat(action):
    return action.replace('.', '')

def process_action_outputs(data, force_rerun=False):
    get_atomic_data(data, force_rerun)
    get_answer_dict(data, force_rerun)

def get_atomic_data(data, force_rerun=False):
    all_actions = set()
    for k,v in data.items():
        if 'new_' in k:
            continue
        if k == 'top_k_output' or k == 'conformal_prediction_output':
            for output_dict in v:
                aciont1 = output_dict['action']
                all_actions.add(action_reformat(aciont1))
                for k2, v2 in output_dict['all_result'].items():
                    action2 = v2['action']
                    all_actions.add(action_reformat(action2))
        elif 'output' in k:
            for output_dict in v:
                all_actions.add(action_reformat(output_dict['action']))
    print(len(all_actions),all_actions)
    atomic_dict = {}
    if 'atomic_dict' in data and not force_rerun:
        atomic_dict = data['atomic_dict']
    
    for action in all_actions:
        if action in atomic_dict and not force_rerun:
            continue
        scene_objects = data['scene_objects'].split(',')
        atomic_action_data = make_action_atomic(action, scene_objects)
        print(atomic_action_data)
        if atomic_action_data['object'].lower() not in data['scene_objects'].lower() and atomic_action_data['object'].lower() not in data['scene'].lower():
            print(f'object: {atomic_action_data["object"]} not in scene_objects: {data["scene_objects"]}')
            for k in atomic_action_data:
                atomic_action_data[k] = 'NULL'
        atomic_dict[action] = atomic_action_data
        print('*', end='')
    print('*'*100)
    data['atomic_dict'] = atomic_dict
    
def get_answer_dict(data, force_rerun=False):
    answer_dict = {}
    atomic_dict = data['atomic_dict']
    classed_atomic_dict = {}
    for k,v in atomic_dict.items():
        obj = v['object'].lower().strip()
        obj = refomat_obj(obj)
        if obj not in classed_atomic_dict:
            classed_atomic_dict[obj] = []
        classed_atomic_dict[obj].append(v['reconstructed_action'])
    # for k,v in classed_atomic_dict.items():
    #     print('*'*100)
    #     print(k)
    #     for action in v:
    #         print(action)
    # print('*'*100)
    all_action_classes = []
    for k,v in classed_atomic_dict.items():
        print(len(v), end='')
        action_classes = classify_similar_actions(v)
        all_action_classes.extend(action_classes)
        print('-', end='')
    action_classes = all_action_classes
    for index, action_class in enumerate(action_classes): 
        print('-'*20+'action_class'+'-'*20)
        for action in action_class:
            print(action)
            for k,v in atomic_dict.items():
                if v['reconstructed_action'] == action:
                    # print(k)
                    if index not in answer_dict:
                        answer_dict[index] = {'reconstructed_action':action}
                        answer_dict[index].update(v)               
                    v['answer'] = index
    data['answer_dict'] = answer_dict
    print('*'*100)

def refomat_obj(obj):
    delete_words = ['bag of', 'box of', 'bottle of', 'a', 'an', 'the'] 
    for word in delete_words:
        obj = obj.replace(word, '')
    while '  ' in obj:
        obj = obj.replace('  ', ' ')
    return obj

# import pickle
# i = 102
# pickle_file_path = "cache/data{}.pkl".format(i)
# with open(pickle_file_path, "rb") as f:
#     data = pickle.load(f)
# get_atomic_data(data, force_rerun=True)
# get_answer_dict(data, force_rerun=True)



# %%
# 正确答案检测
def find_matching_answer(data):
    user_intent_location = data['user_intent_location']
    user_intent_object = data['user_intent_object']
    answer_dict = data['answer_dict']
    print(f'user_intent_location: {user_intent_location}')
    print(f'user_intent_object: {user_intent_object}')
    print(f'answer_dict: {answer_dict}')
    data['right_answer'] = -1
    to_user = False
    if user_intent_location == 'pick-up':
        user_intent_location = ''
        to_user = True
    for answer_index, answer_data in answer_dict.items():
        if not user_intent_location.lower() in answer_data['to_location'].lower():
            continue
        if to_user:
            should_have_elements = ['NULL','user','you','here']
            if not any([element.lower() in answer_data['to_location'].lower() for element in should_have_elements]):
                continue
            # print(f'location: {answer_data["to_location"]}')
        for u_obj in user_intent_object.split(','):
            if 'with' in u_obj:
                u_obj = u_obj.split('with')[0].strip()
            if u_obj.lower() in answer_data['object'].lower():
                print(f'right answer: {answer_data["reconstructed_action"]}')
                data['right_answer'] = answer_index

# import pickle
# i = 216
# pickle_file_path = "cache/data{}.pkl".format(i)
# with open(pickle_file_path, "rb") as f:
#     data = pickle.load(f)
# find_matching_answer(data)
# data['answer_dict']


# %%
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
def all_process_method(data):
    # 收集不同测试方法数据
    # print('collect_vanilla_outputs'+'-'*100)
    # collect_vanilla_outputs(prompt_vanilla, data)
    # print('collect_cot_output'+'-'*100)
    # collect_cot_output(data)
    # print('collect_self_prob_outputs'+'-'*100)
    # collect_self_prob_outputs(data)
    # print('collect_multi_step_output'+'-'*100)
    # collect_multi_step_output(data)
    # print('collect_top_k_outputs'+'-'*100)
    # collect_top_k_outputs(data)
    print('collect_conformal_outputs'+'-'*100)
    collect_conformal_outputs(data)


    # 对齐动作（重复归一）
    print('process_action_outputs'+'-'*100)
    process_action_outputs(data)
    # 找到正确答案
    print('find_matching_answer'+'-'*100)
    find_matching_answer(data)
    #动作更新为answer index
    print('align_atomic_actions'+'-'*100)
    align_atomic_actions(data)


# %%

if TEST_FLAG:
    scenario_data = scenario_data[:5]

for data_index in tqdm.tqdm(range(len(scenario_data))):
    data = scenario_data[data_index]
    pickle_path = f"./cache/data{data['index']}.pkl"

    if os.path.exists(pickle_path):
        # # --------- 为了重新运行其中一些内容 还有force_rerun参数记得改
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        scenario_data[data_index] = data
        # # ---------- 结束
        # continue
    # error_path = f'./cache/error{data["index"]}.pkl'
    # if os.path.exists(error_path):
    #     continue
    print('-'*20, f'开始处理新情景[{data["index"]}]', '-'*20)
    try:
        all_process_method(data)
        # save data to ./cache/data[index].pkl
        with open(f'./cache/data{data["index"]}.pkl', 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        # raise e
        with open(f'./cache/error{data["index"]}.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(e)
        continue

# %%
# 下一步去cache_check文件运行

# %%
scenario_data[0]

# %%



