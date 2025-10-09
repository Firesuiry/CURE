# %%
import os
TEST_FLAG = os.getenv('TEST_FLAG', '0') == '1'
cache_dir = os.getenv('CACHE_DIR', '0')
if cache_dir == "0":
    cache_dir = None

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor
from transformers import LogitsProcessorList, LogitsProcessor
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
class LLAMA:
    def __init__(self, model_name):
        self.device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto",cache_dir=cache_dir, load_in_8bit=True)
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
llama_obj = LLAMA('unsloth/Llama-3.3-70B-Instruct')

# %%
with open('dataset_generate/objects.txt','r') as f:
    objects = f.read().split('\n')
object_dict = {}
object_type_dict = {}
for obj in objects:
    obj = obj.replace("'","")
    if obj.startswith('==='):
        key = obj.strip().replace('=','')
        object_dict[key] = []
    else:
        object_dict[key].append(obj.strip())
        object_type_dict[obj.strip()] = key

# %%
import random

from numpy import real
from torch import rand
kitchenwares = object_dict['kitchenware']
others = []
for key in object_dict:
    if key != 'kitchenwares':
        others.extend(object_dict[key])




def chat(scence, real_want_object, prompt):
    prompt = prompt.format(scence=scence, real_want_object=real_want_object)
    device = llama_obj.device
    model = llama_obj.model 
    tokenizer = llama_obj.tokenizer
    messages = [
        {"role": "system", "content": "You are a kindly person."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    decode_outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    print(decode_outputs[0])
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        return_dict_in_generate=True, 
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
    # 将output中的input删除，只保留新生成的output
    new_generate_sequence = outputs.sequences[0, input_ids.size(1):-1]
    decoded_output = tokenizer.decode(new_generate_sequence)
    print(decoded_output)
    return decoded_output

def bring_task_generate0():
    # random choose 3 objects from others
    want_objects = random.sample(others,3)
    scence = ', '.join(want_objects)
    real_want_object = random.choice(want_objects)
    prompt = '''
On the counter, there is {scence}. You want {real_want_object}, but you need to be more tactful and not directly say what you want. What should you say? Please say it directly, without providing any other information.
Here is an example: there is a Pepsi, a Coke, and a RedBull， You Want RedBull, saying Bring me that caffeinated drink.'''
    response = chat(scence, real_want_object, prompt)
    '''
    Scene: a Pepsi, a bottled unsweetened tea, and a RedBull
    Task: Put that bottled unsweetened tea in the bottom drawer.
    User intent (object): bottled unsweetened tea
    User intent (location): bottom drawer
    Possible objects: bottled unsweetened tea
    Possible locations: bottom drawer
    Scene objects: Pepsi, bottled unsweetened tea, RedBull
    Task category: ambiguous_task'''
    task_dict = {
        'Scene':scence,
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':'',
        'Possible objects':scence,
        'Possible locations':'',
        'Scene objects': scence,
        'Task category':'bring_task0',
        'Ambiguous':False,
    }
    return task_dict

# need choose item from 3 drink
def bring_task_generate_juice_favor():
    # random choose 3 objects from juice
    want_objects = random.sample(object_dict['juice'],3)
    scence = ', '.join(want_objects)
    favorite_juices = []
    for i in range(3):
        if random.random() > 0.4:
            favorite_juices.append(want_objects[i])
    real_want_object = ','.join(favorite_juices)
    response = 'Please give me my favorite drink.'
    task_dict = {
        'Scene':scence,
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':'',
        'Possible objects':scence,
        'Possible locations':'',
        'Scene objects': scence,
        'Task category':'bring_task_juice_favor',
        'Ambiguous':True,
    }
    return task_dict

def bring_task_generate_juice_favor2():
    want_objects = random.sample(object_dict['juice'],3)
    scence = ', '.join(want_objects)
    favorite_juices = []
    real_want_object = random.choice(want_objects)
    prompt = '''
On the counter, there is {scence}. You want {real_want_object}, but you need to be more tactful and NOT directly say what you want. What should you say? Please say it directly, without providing any other information.
Here is an example: there is a Pepsi, a Coke, and a RedBull， You Want RedBull, saying Bring me that caffeinated drink.'''
    response = chat(scence, real_want_object, prompt)
    '''
    Scene: a Pepsi, a bottled unsweetened tea, and a RedBull
    Task: Put that bottled unsweetened tea in the bottom drawer.
    User intent (object): bottled unsweetened tea
    User intent (location): bottom drawer
    Possible objects: bottled unsweetened tea
    Possible locations: bottom drawer
    Scene objects: Pepsi, bottled unsweetened tea, RedBull
    Task category: unambiguous_task'''
    task_dict = {
        'Scene':scence,
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':'',
        'Possible objects':scence,
        'Possible locations':'',
        'Scene objects': scence,
        'Task category':'bring_task_juice_favor2',
        'Ambiguous':False,
    }
    return task_dict




def throw_task_generate():
    throw_obj_class = random.choice(['juice','food','fruit', 'kitchenware'])
    # 扔掉喝一半的饮料
    target_num = random.randint(1,2)
    want_objects = random.sample(object_dict[throw_obj_class],target_num)
    all_other_objs = []
    for key in object_dict:
        if key != throw_obj_class:
            all_other_objs.extend(object_dict[key])
    other_objects = random.sample(all_other_objs,3-target_num) if target_num < 3 else []
    scence = ', '.join(want_objects + other_objects)
    real_want_object = random.choice(want_objects)
    if throw_obj_class == 'juice': 
        intent_location = 'recycling' 
        response = "I don't want to drink anymore, please throw away that drink."
    if throw_obj_class == 'food': 
        intent_location = 'landfill'
        response = "I ate it already. Can you dispose of it?" 
    if throw_obj_class == 'fruit': 
        intent_location = 'compost'
        response = "It is rotten. Please throw it away."
    if throw_obj_class == 'kitchenware':
        intent_location = 'landfill'
        response = "I don't want to use it any more. Can you please dispose of it?"
    task_dict = {
        'Scene':scence,
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':intent_location,
        'Possible objects':scence,
        'Possible locations':'',
        'Scene objects': scence,
        'Task category':'throw_task',
        'Ambiguous': target_num == 2,
    }
    return task_dict

def put_to_somewhere_task_generate():
    # 1. 放到瓶装饮料旁边
    # 2. 放到水果旁边
    # 3. 放到食物旁边
    # 4. 放到薯片旁边
    # 5. 放到特定物品旁边

    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    # 随机抽取一个要操作的物品
    want_objects = random.sample(others,1)
    # 随机抽取任务类型
    task_type = random.choices(['bottle','food','fruit','chip','specific'],probs)[0]
    # 随机生成目标数量
    target_num = random.randint(1,2)
    if task_type == 'specific':
        target_num = 2
    # 生成目标数据集和非目标数据集
    all_target_objs = []
    all_other_objs = []
    
    if task_type == 'bottle':
        for obj in object_dict['juice']:
            if 'bottle' in obj:
                all_target_objs.append(obj)
            else:
                all_other_objs.append(obj)
        for key in object_dict:
            if key != 'juice':
                all_other_objs.extend(object_dict[key])
    elif task_type == 'food':
        all_target_objs = object_dict['food']
        for key in object_dict:
            if key != 'food':
                all_other_objs.extend(object_dict[key])
    elif task_type == 'fruit':
        all_target_objs = object_dict['fruit']
        for key in object_dict:
            if key != 'fruit':
                all_other_objs.extend(object_dict[key])
    elif task_type == 'chip':
        for obj in object_dict['food']:
            if 'chip' in obj:
                all_target_objs.append(obj)
            else:
                all_other_objs.append(obj)
        for key in object_dict:
            if key != 'food':
                all_other_objs.extend(object_dict[key])
    elif task_type == 'specific':
        for key in object_dict:
            all_target_objs.extend(object_dict[key])
    # 从目标数据集中随机抽取目标
    target = random.sample(all_target_objs,target_num)
    # 从非目标数据集中随机抽取非目标
    other_objects = random.sample(all_other_objs,2-target_num) if target_num < 2 else []
    # 是否模糊
    ambiguous = target_num == 2
    if task_type == 'specific':
        ambiguous = False
            
    # 选择一个目标
    real_want_object = want_objects[0]
    scence_objs = want_objects + target + other_objects
    random.shuffle(scence_objs)
    scence = ', '.join(scence_objs)
    # 生成任务
    if task_type == 'bottle':
        response = 'Put that {want_obj} near the bottle.'
    elif task_type == 'food':
        response = 'Put that {want_obj} near the food.'
    elif task_type == 'fruit':
        response = 'Put that {want_obj} near the fruit.'
    elif task_type == 'chip':
        response = 'Put that {want_obj} near the chips.'
    elif task_type == 'specific':
        response = 'Put that {want_obj} near the {target_obj}.'
    response = response.format(want_obj=real_want_object,target_obj=target[0])
    task_dict = {
        'Scene':scence,
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':target[0],
        'Possible objects':scence,
        'Possible locations':', '.join(target),
        'Scene objects': scence,
        'Task category':'put_to_somewhere_task[{}]'.format(task_type),
        'Ambiguous':ambiguous,
    }
    return task_dict

def clean_dirty_task_generate():
    obj = random.choice(object_dict['kitchenware'])
    target = [random.choice(others)]
    clean_obj = dirty_obj = ''
    if obj.startswith('a '):
        clean_obj = 'a clean ' + obj[2:]
        dirty_obj = 'a dirty ' + obj[2:]
    elif obj.startswith('an '):
        clean_obj = 'a clean ' + obj[3:]
        dirty_obj = 'a dirty ' + obj[3:]
    else:
        clean_obj = 'clean ' + obj
        dirty_obj = 'dirty ' + obj

    target.append(clean_obj)
    target.append(dirty_obj)
    random.shuffle(target)
    scence = ', '.join(target)
    real_want_object = clean_obj
    response = f'Please give me {obj}.'
    task_dict = {
        'Scene':scence,
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':'',
        'Possible objects':scence,
        'Possible locations':'',
        'Scene objects': scence,
        'Task category':'clean_dirty_task',
        'Ambiguous':False,
    }
    return task_dict

def bring_task_mulit_objects1():
    task_type = random.choice(['juice','food','fruit'])
    ambiguous = random.random() > 0.5
    if ambiguous:
        target_num = random.randint(2, 3)
    else:
        target_num = 1
    # 生成目标数据集和非目标数据集
    all_target_objs = []
    all_other_objs = []
    all_target_objs = object_dict[task_type]
    for key in object_dict:
        if key != task_type:
            all_other_objs.extend(object_dict[key])
    # 从目标数据集中随机抽取目标
    want_objects = random.sample(all_target_objs,target_num)
    # 从非目标数据集中随机抽取非目标
    other_objects = random.sample(all_other_objs,3-target_num) if target_num < 3 else []
    all_scence_objs = want_objects + other_objects
    real_want_object = want_objects[0]
    random.shuffle(all_scence_objs)
    scence = ', '.join(all_scence_objs)
    if task_type == 'juice':
        response = 'Please give me that drink.'
    if task_type == 'food':
        response = 'Please give me that food.'
    if task_type == 'fruit':
        response = 'Please give me that fruit.'
    task_dict = {
        'Scene':scence,
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':'',
        'Possible objects':scence,
        'Possible locations':'',
        'Scene objects': scence,
        'Task category':'bring_task_mulit_objects1',
        'Ambiguous':ambiguous,
    }
    return task_dict

def bring_task_mulit_objects2():
    task_type = random.choice(['chip', 'bottle',])
    ambiguous = random.random() > 0.5
    if ambiguous:
        target_num = random.randint(2, 3)
    else:
        target_num = 1
    # 生成目标数据集和非目标数据集
    all_target_objs = []
    all_other_objs = []
    for obj in others:
        if task_type in obj:
            all_target_objs.append(obj)
        else:
            all_other_objs.append(obj)
    # 从目标数据集中随机抽取目标
    want_objects = random.sample(all_target_objs,target_num)
    # 从非目标数据集中随机抽取非目标
    other_objects = random.sample(all_other_objs,3-target_num) if target_num < 3 else []
    all_scence_objs = want_objects + other_objects
    real_want_object = want_objects[0]
    random.shuffle(all_scence_objs)
    scence = ', '.join(all_scence_objs)
    if task_type == 'chip':
        response = 'Please give me that chip.'
    if task_type == 'bottle':
        response = 'Please give me that bottle.'
    task_dict = {
        'Scene':scence,
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':'',
        'Possible objects':scence,
        'Possible locations':'',
        'Scene objects': scence,
        'Task category':'bring_task_mulit_objects2',
        'Ambiguous':ambiguous,
    }
    return task_dict

def put_sth_to_drawer():
    want_objects = random.sample(others,3)
    available_taget = ['top drawer','bottom drawer']
    ambiguous = random.random() > 0.5
    target = random.choice(available_taget)
    target2 = target if not ambiguous else 'drawer'
    response = f'Put that {want_objects[0]} in the {target2}.'
    random.shuffle(want_objects)
    task_dict = {
        'Scene':', '.join(want_objects),
        'Task':response,
        'User intent (object):':random.choice(want_objects),
        'User intent (location):':target,
        'Possible objects':', '.join(want_objects),
        'Possible locations':','.join(available_taget),
        'Scene objects': ', '.join(want_objects),
        'Task category':'put_sth_to_drawer',
        'Ambiguous':ambiguous,
    }
    return task_dict

def get_multi_object(keyword, num):
    def t1(keyword, num):
        want_objects = []
        while len(want_objects) < num:
            for obj in others:
                if keyword in obj and random.random() > 0.3:
                    want_objects.append(obj)
                    if len(want_objects) == num:break
        return want_objects
    
    def t2(keyword, num):
        want_objects = random.sample(object_dict[keyword],num)
        return want_objects
    
    if keyword in object_dict:
        return t2(keyword, num)
    else:
        return t1(keyword, num)
    
def get_multi_object_exclude(keyword, num):
    def t1(keyword, num):
        want_objects = []
        while len(want_objects) < num:
            for obj in others:
                if not keyword in obj and random.random() > 0.3:
                    want_objects.append(obj)
                    if len(want_objects) == num:break
        return want_objects
    
    def t2(keyword, num):
        all_objs = []
        for key in object_dict:
            if key != keyword:
                all_objs.extend(object_dict[key])
        want_objects = random.sample(all_objs,num)
        return want_objects
    
    if keyword in object_dict:
        return t2(keyword, num)
    else:
        return t1(keyword, num)

def put_sth_to_multi_target():
    task_type = random.choice(['can', 'bottle', 'fruit', 'food', 'chip'])
    ambiguous = random.random() > 0.5
    if ambiguous:
        target_num = 2
    else:
        target_num = 1
    target_objects = get_multi_object(task_type,target_num)
    other_objects = get_multi_object_exclude(task_type,2-target_num) if target_num < 2 else []
    real_want_object = random.choice(others)
    
    middle = random.choice(['besides','near','next to'])
    act = random.choice(['put','place','bring'])
    response = f'{act} that {real_want_object} {middle} {task_type}.'
    sence_objs = target_objects + other_objects + [real_want_object]
    random.shuffle(sence_objs)
    task_dict = {
        'Scene':', '.join(sence_objs),
        'Task':response,
        'User intent (object):':real_want_object,
        'User intent (location):':random.choice(target_objects),
        'Possible objects':', '.join(target_objects),
        'Possible locations':', '.join(target_objects),
        'Scene objects': ', '.join(target_objects),
        'Task category':'put_sth_to_multi_target',
        'Ambiguous':ambiguous,
    }
    return task_dict

task_prob_dict = {
    bring_task_generate0:0.1,
    bring_task_generate_juice_favor:0.1,
    bring_task_generate_juice_favor2:0.1,
    throw_task_generate:0.2,
    put_to_somewhere_task_generate:0.4,
    clean_dirty_task_generate:0.2,
    bring_task_mulit_objects1:0.1,
    bring_task_mulit_objects2:0.1,
    put_sth_to_drawer:0.1,
    put_sth_to_multi_target:0.1,
}
all_task = []
task_prob = []
for key in task_prob_dict:
    all_task.append(key)
    task_prob.append(task_prob_dict[key])
    
put_sth_to_multi_target()

# %%
import json
import os
import tqdm.notebook as tqdm
generate_num = 100000
if TEST_FLAG:
    generate_num = 100

for i, task_method in enumerate(all_task):
    task_num = int(generate_num * task_prob[i] / sum(task_prob))
    dir_path = 'dataset_generate/task_data2/{}'.format(task_method.__name__)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for j in tqdm.trange(task_num):
        file_name = 'dataset_generate/task_data2/{}/{}.json'.format(task_method.__name__,j)
        if os.path.exists(file_name):
            continue
        task_dict = task_method()
        assert 'Ambiguous' in task_dict
        with open(file_name,'w') as f:
            f.write(json.dumps(task_dict, indent=1))
        

# %%
import json
import random
from pathlib import Path
p = Path('dataset_generate/task_data2/')
all_task = []
file_num = 0
for f in p.glob('*/*.json'):
# for f in p.glob('*.txt'):
    task_dict = json.loads(f.read_text())
    file_num += 1
    '''
    {'Scene': 'a lemon-lime soda, a root beer, a clean sponge',
    'Task': 'Bring me that citrus-flavored soda.',
    'User intent (object):': 'a lemon-lime soda',
    'User intent (location):': '',
    'Possible objects': 'a lemon-lime soda, a root beer, a clean sponge',
    'Possible locations': '',
    'Scene objects': 'a lemon-lime soda, a root beer, a clean sponge',
    'Task category': 'unambiguous_task'}
    '''
    scene_objects = task_dict['Scene objects'].split(', ')
    user_intent_object = task_dict['User intent (object):']
    user_intent_location = task_dict['User intent (location):']
    if user_intent_object == '':
        # print(task_dict)
        continue
    action_list = []
    if len(list(set(scene_objects))) < len(scene_objects):
        continue
    scene_objs2 = task_dict['Scene'].split(', ')
    if len(list(set(scene_objs2))) < len(scene_objs2):
        continue
    for obj in scene_objects:
        if task_dict['User intent (location):'] == '':
            operate = random.choice(['bring', 'pick-up'])
            user_random = random.choice([' to you', ' to user', ''])
            action = f'{operate} {obj}{user_random}.'
            action_success = 0
            if obj in user_intent_object:
                action_success = 1
            action_list.append([action, action_success])
        elif task_dict['User intent (location):'] in ['recycling', 'landfill', 'compost']:
            for target_location in ['recycling', 'landfill', 'compost']:
                operate = random.choice(['throw', 'move', 'bring', 'pick-up'])
                action = f'{operate} {obj} to {target_location}.'
                action_success = 0
                if obj in user_intent_object and target_location == task_dict['User intent (location):']:
                    action_success = 1
                action_list.append([action, action_success])
        elif 'drawer' in task_dict['User intent (location):']:
            operate = random.choice(['bring', 'pick-up', 'move', 'put'])
            targets = ['top drawer', 'bottom drawer']
            for target in targets:
                action = f'{operate} {obj} to {target}.'
                action_success = 0
                if obj in user_intent_object and target in task_dict['User intent (location):']:
                    action_success = 1
                action_list.append([action, action_success])
        else:
            target_location = obj
            if user_intent_object == target_location:
                continue
            operate = random.choice(['bring', 'pick-up', 'move'])
            action = f'{operate} {user_intent_object} to {target_location}.'
            action_success = 0
            if target_location == user_intent_location:
                action_success = 1
            action_list.append([action, action_success])
    task_dict['Action'] = action_list
    assert sum([a[1] for a in action_list]) > 0, task_dict
    # shuffle the scene objects
    random.shuffle(scene_objects)
    task_dict['Scene objects'] = ', '.join(scene_objects)
    all_task.append(task_dict)
print(f'number of tasks: {len(all_task)} from {file_num} files')
# write to json
if not os.path.exists('task'):
    os.makedirs('task')
with open('dataset_generate/task/task_action2.json', 'w') as f:
    json.dump(all_task, f, indent=1)

# %%



