# %%
# @markdown **Sample from the distribution**

import json
import random
dataset = []
COLORS = {
    'blue':   (78/255,  121/255, 167/255, 255/255),
    'red':    (255/255,  87/255,  89/255, 255/255),
    'green':  (89/255,  169/255,  79/255, 255/255),
    'orange': (242/255, 142/255,  43/255, 255/255),
    'yellow': (237/255, 201/255,  72/255, 255/255),
    'purple': (176/255, 122/255, 161/255, 255/255),
    'pink':   (255/255, 157/255, 167/255, 255/255),
    'cyan':   (118/255, 183/255, 178/255, 255/255),
    'brown':  (156/255, 117/255,  95/255, 255/255),
    'gray':   (186/255, 176/255, 172/255, 255/255),
}
# @markdown **Set up distribution**

blocks = ['green block', 'blue block', 'yellow block']
bowls = ['green bowl', 'blue bowl', 'yellow bowl']

instruction_ambiguities = {
    'put the block in the {color} bowl': 'block color',
    'put the {color} block in the bowl': 'bowl color',
    'put the {color} block in the {color} bowl': None,
    'put the {color} block close to the {color} bowl': 'direction',
    'put the {color} block to the {direction} of the {color} bowl': None
}
instruction_templates = list(instruction_ambiguities.keys())

def same(s1,s2):
    return s1.lower().strip() == s2.lower().strip()
colors = ['green', 'blue', 'yellow']
directions = ['front', 'back', 'left', 'right']

def get_data(blocks, bowls, instruction_ambiguities, instruction_templates, same, colors, directions, i):
    data = {}
    instruction_orig = random.choice(instruction_templates)
    instruction = instruction_orig

    # sample colors if needed
    num_color_in_instruction = instruction.count('{color}')
    if num_color_in_instruction > 0:
        color_instruction = random.choices(colors, k=num_color_in_instruction)
        for color in color_instruction:
            instruction = instruction.replace('{color}', color)

    # sample didrection if needed
    if '{direction}' in instruction:
        direction = random.choice(directions)
        instruction = instruction.replace('{direction}', direction)

    # sample goal based on ambiguities
    ambiguity = instruction_ambiguities[instruction_orig]
    if ambiguity and 'color' in ambiguity:
        true_color = random.choice(colors)
    elif ambiguity and 'direction' in ambiguity:
        true_direction = random.choice(directions)

    # determine the goal in the format of [pick_obj, relation (in, left, right, front, back), target_obj]
    instruction_split = instruction.split()
    block_attr = instruction_split[instruction_split.index('block')-1]
    if 'the' == block_attr:  # ambiguous
        pick_obj = true_color + ' block'
    else:
        pick_obj = block_attr + ' block'
    bowl_attr = instruction_split[instruction_split.index('bowl')-1]
    if 'the' == bowl_attr:  # ambiguous
        target_obj = true_color + ' bowl'
    else:
        target_obj = bowl_attr + ' bowl'
    if 'close to' in instruction:
        relation = true_direction
    elif 'in' in instruction:
        relation = 'in'
    else:
        relation = instruction_split[instruction_split.index(
            'of')-1]  # bit hacky

    # fill in data
    data['index'] = i
    data['environment'] = blocks + bowls  # fixed set
    data['instruction'] = instruction
    assert '{' not in instruction,instruction
    data['goal'] = [pick_obj, relation, target_obj]
    data['ambiguous'] = ambiguity
    # generate a false goal
    pick_obj_false = ''
    target_obj_false = ''
    relation_false = ''
    while pick_obj_false == '':
        pick_obj_false = random.choice(blocks)
        if same(pick_obj_false, pick_obj):
            pick_obj_false = ''
    while target_obj_false == '':
        target_obj_false = random.choice(bowls)
        if same(target_obj_false, target_obj):
            target_obj_false = ''
    while relation_false == '':
        relation_false = random.choice(['in', 'left', 'right', 'front', 'back'])
        if same(relation_false, relation):
            relation_false = ''
    
    data['false_goal'] = [pick_obj_false, relation_false, target_obj_false]
    return data

dataset = []
for i in range(300):
    data = get_data(blocks, bowls, instruction_ambiguities, instruction_templates, same, colors, directions, i)
    dataset.append(data)
with open('task/dataset_val.json', 'w') as f:
    json.dump(dataset, f)
    
dataset = []
for i in range(10000):
    data = get_data(blocks, bowls, instruction_ambiguities, instruction_templates, same, colors, directions, i)
    dataset.append(data)
with open('task/dataset_train.json', 'w') as f:
    json.dump(dataset, f)

# print a few
print('Showing the first five sampled scenarios')
print('Environment:', dataset[0]['environment'])
for i in range(5):
    data = dataset[i]
    print(f'==== {i} ====')
    print('Instruction:', data['instruction'])
    print('Goal (pick_obj, relation, target_obj):', data['goal'])

# %%



