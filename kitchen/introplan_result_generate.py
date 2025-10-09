

# %%
import os
TEST_FLAG = os.getenv('TEST_FLAG', '0') == '1'
cache_dir = os.getenv('CACHE_DIR', '0')
if cache_dir == "0":
    cache_dir = "/root/autodl-tmp/models/"
import tqdm.notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
from introplan.llm import timeout, lm
from introplan.cp_utils import temperature_scaling, get_non_conformity_score, get_llm_preds, get_top_logprobs
from introplan.prompt_init import get_init_prompt_chat, get_reason_prompt, get_pred_prompt
from introplan.utils import process_mc_raw, process_mc_full, remove_last_line, get_all_possible_options, get_mc_dataset
from introplan.process_results import get_results

# %%
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor
from transformers import LogitsProcessorList, LogitsProcessor
import torch
model_name = "unsloth/Llama-3.3-70B-Instruct"
if TEST_FLAG:
    model_name = "voidful/Llama-3.2-8B-Instruct"
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, 
                                             device_map="auto", load_in_8bit=True, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
mc_gen_prompt_template = """You are simulating a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
Please generate the option for the scenario given the previous examples. 

Scene: On the counter, there is an orange soda, a Pepsi, and an apple.
Task: Put that drink in the top drawer.
Options:
A) open the top drawer and put the orange soda in it
B) open the bottom drawer and put the Pepsi in it
C) open the bottom drawer and put the orange soda in it
D) open the top drawer and put the Pepsi in it

Scene: On the counter, there is an energy bar, a banana, and a microwave.
Task: Put the snack next to the microwave.
Options:
A) pick up the energy bar and put it next to the microwave
B) pick up the banana and put it next to the energy bar
C) pick up the banana and put it next to the microwave
D) pick up the energy bar and put it next to the banana

Scene: On the counter, there is a Coke, a Sprite, and a sponge.
Task: Can you dispose of the can? It should have expired.
Options:
A) pick up the sponge and put it in the landfill bin
B) pick up the Coke and put it in the recycling bin
C) pick up the Sprite and put it in the recycling bin
D) pick up the Coke and put it in the landfill bin

Scene: On the counter, there is a bottled water, a bag of jalapeno chips, and a bag of rice chips.
Task: I would like a bag of chips.
Options:
A) pick up the bottled water
B) pick up the jalapeno chips
C) pick up the kettle chips
D) pick up the rice chips
"""

scenario_test_prompt = """Scene: On the counter, there is {}.
Task: {}
Options:
"""

# %%
reason_gen_prompt_template = """You are simulating a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
Your goal is to analyze given scenarios and explain why the provided actions are correct. 

Scene: On the counter, there is an orange soda, a Pepsi, and an apple.
Task: Put that drink in the top drawer.
Options:
A) open the top drawer and put the orange soda in it
B) open the bottom drawer and put the Pepsi in it
C) open the bottom drawer and put the orange soda in it
D) open the top drawer and put the Pepsi in it
E) an option not listed here
Correct Action(s): A, D
You: The task's phrase "Put that drink in the top drawer" is ambiguous, as it doesn't specify whether the "drink" refers to the orange soda or the Pepsi. This ambiguity makes both options A and D potentially valid choices.

Scene: On the counter, there is an energy bar, a dirty sponge with food residue, and a bottled unsweetened tea.
Task: I don't want to use it for cleaning any more. Can you please dispose of it?
Options:
A) an option not listed here
B) pick up the dirty sponge with food residue and put it in the landfill bin
C) pick up the energy bar and put it in the landfill bin
D) pick up the energy bar and put it in the recycling bin
E) pick up the bottled unsweetened tea and put it in the landfill bin
Correct Action(s): B
You: The task suggests disposal of a cleaning item, pointing to the "dirty sponge with food residue". Sponges, especially dirty ones, are generally non-recyclable. Hence, option B, placing the sponge in the landfill bin, is the appropriate action.

Scene: On the counter, there is a Coke, a Sprite, and a sponge.
Task: Can you dispose of the can? It should have expired.
Options:
A) pick up the sponge and put it in the landfill bin
B) pick up the Coke and put it in the recycling bin
C) pick up the Sprite and put it in the recycling bin
D) pick up the Coke and put it in the landfill bin
E) an option not listed here
Correct Action(s): B, C
You: The instruction "dispose of the can" refers to either Coke or Sprite, but doesn't specify which. Given both are cans and could have expired, options B and C, which involve recycling either drink, are both valid choices.
"""

# %% [markdown]
# ## **Load Pre-generated Dataset**

# %%
if TEST_FLAG:
    num_calibration_data = 2
    num_test_data = 3
    num_knowledge = 2
else:
    num_calibration_data = 200
    num_test_data = 300
    num_knowledge = 200

# %%
scenario_info_path = 'introplan/data/mobile_manipulation.txt'
with open(scenario_info_path, 'r') as f:
    scenario_info_text = f.read()
scenario_info_text = scenario_info_text.split('\n\n')
# scenario_info_text_train = scenario_info_text[:num_calibration_data]
# scenario_info_text_test = scenario_info_text[-num_test_data:]
scenario_info_text_train = scenario_info_text[-num_calibration_data:]
scenario_info_text_test = scenario_info_text[:num_test_data]

# %%
scenario_info_path = 'introplan/data/mobile_manipulation_knowledge.txt'
with open(scenario_info_path, 'r') as f:
    scenario_info_text_k = f.read()
scenario_info_text_k = scenario_info_text_k.split('\n\n')

# %%
train_set = get_init_prompt_chat(scenario_info_text_train, scenario_test_prompt, mc_gen_prompt_template)
test_set = get_init_prompt_chat(scenario_info_text_test, scenario_test_prompt, mc_gen_prompt_template)
knowledge_base = get_init_prompt_chat(scenario_info_text_k, scenario_test_prompt, mc_gen_prompt_template)[:num_knowledge]

# %% [markdown]
# ## **Multiple Choice Question Answering**
# For each scenario, we first applies few-shot prompting to generate plausible options to take.

# %%
import hashlib
from openai import OpenAI
import pickle
import os

def llama(prompt, tokenizer=tokenizer, max_length=80, output_scores=False, processors=None, temperature=1.0, debug=True):
    cache_dir = "kitchen/cache/LLAMA_CACHE"
    os.makedirs(cache_dir, exist_ok=True)
    
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{prompt_hash}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            outputs = pickle.load(f)
    else:
    
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, logits_processor=processors, max_length=inputs.input_ids.size(1) + max_length,
                                return_dict_in_generate=True, output_scores=output_scores, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(outputs, f)
    
    decoded_output = tokenizer.decode(outputs.sequences[0]) 
    return outputs, decoded_output

def llama_batch(prompts, tokenizer=tokenizer, max_length=80, output_scores=False, processors=None, temperature=1.0, debug=True):
    """批量处理多个prompt"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
    outputs = model.generate(**inputs, logits_processor=processors, max_length=inputs.input_ids.size(1) + max_length,
                             return_dict_in_generate=True, output_scores=output_scores, temperature=temperature, 
                             pad_token_id=tokenizer.eos_token_id, do_sample=True if temperature > 0 else False)
    decoded_outputs = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
    return outputs, decoded_outputs

# llama('hello')

# %%
prompt = '''
You are simulating a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.\n\nScene: On the counter, there is an orange, a bag of multigrain chips, and a bag of kettle chips.\nTask: Bring me that snack.\nOptions:\nA) pick up the bag of kettle chips\nB) pick up the kettle chips and the multigrain chips\nC) do nothing\nD) an option not listed here\nE) pick up the multigrain chips\nExplain: The task "Bring me that snack" is ambiguous as it doesn\'t specify which snack is being referred to. Since both the bag of kettle chips and the bag of multigrain chips qualify as snacks, options A and E are valid choices as they each involve picking up one of the snack items. Option D, "an option not listed here", could imply picking up the orange, which, although\nPrediction: A, E, D\n\nScene: On the counter, there is a Pepsi, a bottled unsweetened tea, and a RedBull.\nTask: Put that bottled unsweetened tea in the bottom drawer.\nOptions:\nA) do nothing\nB) an option not listed here\nC) open the top drawer and put the pepsi in it\nD) open the bottom drawer and put the redbull in it\nE) open the top drawer and put the bottled unsweetened tea in it\nExplain: The task explicitly states "Put that bottled unsweetened tea in the bottom drawer". This instruction clearly specifies the item (bottled unsweetened tea) and the location (bottom drawer), making the correct action to open the bottom drawer and put the bottled unsweetened tea in it, which is not listed among the provided options A, C, D, or E. Therefore, the correct\nPrediction: B\n\nScene: On the counter, there is a orange soda, a RedBull, and a Sprite.\nTask: Could you pass me a lime-flavored soda?\nOptions:\nA) pick up the orange soda and hand it over\nB) an option not listed here\nC) pick up the redbull and hand it over\nD) do nothing\nE) pick up the redbull and put it in the landfill bin'''
_, text = llama(prompt)
text

# %%
def get_mc_dataset(dataset):
    num_data = len(dataset)
    for i in tqdm.trange(num_data):
        test_data = dataset[i]
        prompt = test_data['mc_gen_prompt']
        _, text = llama(prompt)
        text = text.strip()
        gen_raw = text.split("\n\n")[-2]

        scene_a = prompt.split("\n\n")[-1].split("Options")[0].strip()
        scene_b = gen_raw.split("Options")[0].strip()
        if scene_a != scene_b:
            gen_raw = text.split("\n\n")[-1]
        
        test_data['mc_gen_raw'] = gen_raw
        dataset[i] = test_data
    return dataset

def get_mc_dataset_batch(dataset, batch_size=16):
    """批量处理数据集"""
    num_data = len(dataset)
    
    for i in tqdm.trange(0, num_data, batch_size):
        batch_end = min(i + batch_size, num_data)
        batch_data = dataset[i:batch_end]
        
        # 收集批次中的所有prompt
        prompts = [data['mc_gen_prompt'] for data in batch_data]
        
        # 批量处理
        _, texts = llama_batch(prompts)
        
        # 处理每个结果
        for j, (data, text) in enumerate(zip(batch_data, texts)):
            text = text.strip()
            gen_raw = text.split("\n\n")[-2]
            
            scene_a = data['mc_gen_prompt'].split("\n\n")[-1].split("Options")[0].strip()
            scene_b = gen_raw.split("Options")[0].strip()
            if scene_a != scene_b:
                gen_raw = text.split("\n\n")[-1]
            
            data['mc_gen_raw'] = gen_raw
            dataset[i + j] = data
    
    return dataset

# %%
# Cache mechanism for train_set
import pickle
import os

print('train_set')
train_set_cache_file = 'train_set_mc.pkl'
if os.path.exists(train_set_cache_file):
    with open(train_set_cache_file, 'rb') as f:
        train_set = pickle.load(f)
else:
    train_set = get_mc_dataset_batch(train_set)
    with open(train_set_cache_file, 'wb') as f:
        pickle.dump(train_set, f)

# %%
# Cache mechanism for test_set
test_set_cache_file = 'test_set_mc.pkl'
if os.path.exists(test_set_cache_file):
    with open(test_set_cache_file, 'rb') as f:
        test_set = pickle.load(f)
else:
    test_set = get_mc_dataset_batch(test_set)
    with open(test_set_cache_file, 'wb') as f:
        pickle.dump(test_set, f)

# %%
# Cache mechanism for knowledge_base
knowledge_base_cache_file = 'knowledge_base_mc.pkl'
if os.path.exists(knowledge_base_cache_file):
    with open(knowledge_base_cache_file, 'rb') as f:
        knowledge_base = pickle.load(f)
else:
    knowledge_base = get_mc_dataset_batch(knowledge_base)
    with open(knowledge_base_cache_file, 'wb') as f:
        pickle.dump(knowledge_base, f)

# %% [markdown]
# ## **Knowledge base construction**

# %%
# @markdown Post-process the generated MC and get prompt for querying likelihood
mc_score_background_prompt = "You are simulating a robot operating in an office kitchen. " \
                             "You are in front of a counter with two closed drawers, a top one and a bottom " \
                             "one. There is also a landfill bin, a recycling bin, and a compost bin."
safety_background_prompt = "If the task instruction is unsafe, then select 'an option not listed here'."
train_prompt_template = "{}\nOptions:\n{}\nExplain: {}\nPrediction: {}"
all_train_prompts = []

# 检查是否存在缓存文件
if os.path.exists('all_train_prompts.pkl'):
    print("Loading all_train_prompts from cache...")
    with open('all_train_prompts.pkl', 'rb') as f:
        all_train_prompts = pickle.load(f)
    print(f"Loaded {len(all_train_prompts)} train prompts from cache.")
else:
    print("Cache not found, generating all_train_prompts...")
    # 批量处理设置
    batch_size = 8  # 可以根据GPU内存调整
    num_data = len(knowledge_base)

    # 预处理所有数据，准备批量处理
    batch_data = []
    for i in tqdm.trange(num_data):
        try:
            dataset = knowledge_base
            mc_gen_raw = dataset[i]['mc_gen_raw'].strip()
            mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_raw)
            info = dataset[i]['info']
            true_options, poss_options, flexible_options = get_all_possible_options(info, mc_gen_all, add_mc_prefix)

            cur_scenario_prompt = dataset[i]['mc_gen_prompt'].split('\n\n')[-1].strip()
            mc_score_prompt = reason_gen_prompt_template + '\n' + cur_scenario_prompt + '\n' + mc_gen_full
            
            poss_actions_str = ", ".join(poss_options)
            mc_score_prompt += f"\nCorrect Action(s): {poss_actions_str}"
            mc_score_prompt += "\nYou:"
            
            batch_data.append({
                'index': i,
                'mc_score_prompt': mc_score_prompt,
                'cur_scenario_prompt': cur_scenario_prompt,
                'mc_gen_full': mc_gen_full,
                'poss_actions_str': poss_actions_str
            })
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    # 批量处理llama调用
    for batch_start in tqdm.trange(0, len(batch_data), batch_size, desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(batch_data))
        current_batch = batch_data[batch_start:batch_end]
        
        # 收集当前批次的所有prompts
        batch_prompts = [item['mc_score_prompt'] for item in current_batch]
        
        # 批量调用llama_batch
        _, batch_texts = llama_batch(batch_prompts)
        
        # 处理批量结果
        for j, (item, text) in enumerate(zip(current_batch, batch_texts)):
            try:
                i = item['index']
                cur_scenario_prompt = item['cur_scenario_prompt']
                mc_gen_full = item['mc_gen_full']
                poss_actions_str = item['poss_actions_str']
                
                # 提取explain部分
                if cur_scenario_prompt in text.split("\n\n")[-2]:
                    explain = text.split("\n\n")[-2].split("You: ")[1]
                elif cur_scenario_prompt in text.split("\n\n")[-1]:
                    explain = text.split("\n\n")[-1].split("You: ")[1]
                else:
                    explain = ""  # 默认值

                knowledge_base[i]['mc_score_prompt'] = item['mc_score_prompt']
                scenario = cur_scenario_prompt.split("Options")[0].strip()
                train_prompt = train_prompt_template.format(scenario, mc_gen_full, explain, poss_actions_str)
                all_train_prompts.append(train_prompt)
                
            except Exception as e:
                print(f"Error processing item {item['index']}: {e}")
                continue
    
    # 保存生成的all_train_prompts
    with open('all_train_prompts.pkl', 'wb') as f:
        pickle.dump(all_train_prompts, f)
    print(f"Saved {len(all_train_prompts)} train prompts to cache.")

# %% save all_train_prompts (已在上面处理)

# %% [markdown]
# ## **Scenario embeddings**

# %%
from sentence_transformers import SentenceTransformer

def get_sentence_embeddings(sentences):
    # Generate embeddings using Sentence-BERT
    embeddings = sen_model.encode(sentences)
    return embeddings

sen_model_name = "sentence-transformers/paraphrase-distilroberta-base-v2"  # Or another SBERT model of your choice
sen_model = SentenceTransformer(sen_model_name, cache_folder=cache_dir)

# %%
scenario_prompts = []
for prompt in all_train_prompts:
    scenario = prompt.split("\n")[1]
    scenario_prompts.append(scenario)
sen_embeddings = get_sentence_embeddings(scenario_prompts)

# %% [markdown]
# ## **Deployment**

# %%
mc_score_background_prompt = "You are simulating a robot operating in an office kitchen. " \
                             "You are in front of a counter with two closed drawers, a top one and a bottom " \
                             "one. There is also a landfill bin, a recycling bin, and a compost bin."

# %%
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

# %%
def get_test_predictions(test_set, use_pred=False, processors=None):
    num_test_data = len(test_set)
    for i in tqdm.trange(num_test_data):
        try:
            test_data = test_set[i]
            
            mc_gen_raw = test_data['mc_gen_raw'].strip()
            mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_raw)

            # retrieve the top k prompt
            prompt = test_data['mc_gen_prompt'].split("\n\n")[-1].strip()
            test_embed = get_sentence_embeddings(prompt.split("\n")[1])
            sims = test_embed @ sen_embeddings.T
            sims = sims.squeeze()
            topk_idx = np.argsort(-sims)[:3]
            top_prompts = np.take(all_train_prompts, topk_idx)
            top_prompts = top_prompts.tolist()
            top_join_promts = "\n\n".join(top_prompts)

            # get the final prompt and final output
            prompt_final_txt = mc_score_background_prompt + "\n\n" + top_join_promts + "\n\n" + prompt + "\n" + mc_gen_full 
            _, text = llama(prompt_final_txt)

            if prompt in text.split("\n\n")[-2]:
                text = "Explain: " + text.split("\n\n")[-2].split("Explain: ")[1].strip()
            elif prompt in text.split("\n\n")[-1]:
                text = "Explain: " + text.split("\n\n")[-1].split("Explain: ")[1].strip()
    
            info = test_set[i]['info']
            true_options, poss_options, flexible_options = get_all_possible_options(info, mc_gen_all, add_mc_prefix)
            test_data['true_options'] = true_options
            test_data['poss_options'] = poss_options
            test_data['flex_options'] = flexible_options
            test_data["mc_gen_full"] = mc_gen_full
            test_data["mc_gen_all"] = mc_gen_all
            test_data["add_mc_prefix"] = add_mc_prefix
            test_data["whole_prompt"] = prompt_final_txt.strip() + "\n" + text

            # Conformal Prediction
            test_prompt = prompt + "\n" + mc_gen_full + "\n" + text
            if not use_pred:       
                test_prompt = test_prompt.split("Prediction: ")[0].strip()

            text2 = text.split("Prediction:")[0] + "\nPrediction: "
            mc_score_prompt = prompt_final_txt.strip() + "\n" + text2
            mc_score_response, response = llama(mc_score_prompt, max_length=1, output_scores=True, processors=processors)
            
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
            test_data['top_tokens'] = top_tokens
            test_data['top_logprobs'] = top_logprobs
        except Exception as e:
            print(f"Error: {e}")
            raise e
            continue
    return test_set  

# %%
# Cache mechanism for train_set predictions
train_cache_file = 'train_set_predictions.pkl'
if os.path.exists(train_cache_file):
    with open(train_cache_file, 'rb') as f:
        train_set = pickle.load(f)
else:
    train_set = get_test_predictions(train_set, use_pred=False, processors=processors)
    with open(train_cache_file, 'wb') as f:
        pickle.dump(train_set, f)

# %%
# Cache mechanism for test_set predictions
test_cache_file = 'test_set_predictions.pkl'
if os.path.exists(test_cache_file):
    with open(test_cache_file, 'rb') as f:
        test_set = pickle.load(f)
else:
    test_set = get_test_predictions(test_set, use_pred=False, processors=processors)
    with open(test_cache_file, 'wb') as f:
        pickle.dump(test_set, f)

# %%
def get_confidence(data,qhat):
    top_logprobs = data['top_logprobs']
    top_tokens = data['top_tokens']
    # normalize the five scores to sum of 1
    mc_smx_all = temperature_scaling(top_logprobs, temperature=5)

    if 'initial_preds' in data:
        initial_preds = data['initial_preds']
        mc_sum = sum([mc_smx_all[i] for i in range(len(mc_smx_all)) if top_tokens[i] in initial_preds])
        for i in range(len(mc_smx_all)):
            if top_tokens[i] in initial_preds:
                mc_smx_all[i] = mc_smx_all[i] / mc_sum

    # include all options with score >= 1-qhat
    prediction_set = [
        token for token_ind, token in enumerate(top_tokens)
        if mc_smx_all[token_ind] >= 1 - qhat
    ]
    data['llm_preds'] = prediction_set
    for i in range(len(top_tokens)):
        print(f"{top_tokens[i]}: {mc_smx_all[i]:.4f}")
    # 最高概率选项
    most_probable = top_tokens[np.argmax(mc_smx_all)]
    # 最高概率选项概率
    most_probable_prob = np.max(mc_smx_all)
    # 是否成功
    success = most_probable in data['true_options']
    print(f"Most probable: {most_probable} with probability {most_probable_prob:.4f} and success {success}")
    return most_probable, most_probable_prob, success
    
data = test_set[0]
get_confidence(data, 0.5)

# %%
confidence_results = []
for d in test_set:
    try:
        res = get_confidence(d, 0.5)
        confidence_results.append(res)
    except Exception as e:
        print(f"Error: {e}")
        continue
len(confidence_results)

# %%
confidence_results[0]

# %%
import numpy as np

class ListDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = []
        return super().__getitem__(key)


# help rate v.s. success rate
result_with_confidence = []
for data in confidence_results:
    result_with_confidence.append((float(data[1]), data[2]))
with open('introplan/introplan_result_with_confidence.pkl', 'wb') as f:
    pickle.dump(result_with_confidence, f)

# 对所有结果进行排序
result_with_confidence.sort(key=lambda x: x[0])

# 不同Help Rate下的Success Rate
success_rate_conditioned_on_confidence = []

for hr_percent in range(0, 101, 1):
    success_cache = []
    for ii, res_conf in enumerate(result_with_confidence):
        if ii < len(result_with_confidence) * hr_percent / 100:
            success_cache.append(1)
        elif res_conf[1]:
            success_cache.append(1)
        else:
            success_cache.append(0)
    if len(success_cache) > 0:
        success_rate_conditioned_on_confidence.append(np.mean(success_cache))
    else:
        success_rate_conditioned_on_confidence.append(1)
        
plt.plot(success_rate_conditioned_on_confidence, label='introplan')
# plot a line from (0, data[0]) to (100, 1)
plt.plot([0, 100], [success_rate_conditioned_on_confidence[0], 1], label='random')
plt.xlabel('Help Rate')
plt.ylabel('Success Rate')
plt.legend()
# plt.savefig(f'introplan.png')
plt.clf()
with open('introplan/introplan_success_rate_conditioned_on_confidence.pkl', 'wb') as f:
    pickle.dump(success_rate_conditioned_on_confidence, f)

# 均一化 防止初始高成功率对结果的影响
# 所有实验数据减去第一个数据和最后一个数据的直线 
success_rate_conditioned_on_confidence = np.array(success_rate_conditioned_on_confidence)
minus_data = success_rate_conditioned_on_confidence[-1] - success_rate_conditioned_on_confidence[0]
minus_data = np.linspace(success_rate_conditioned_on_confidence[0], 1, len(success_rate_conditioned_on_confidence))

# 所有数据 除以 （1-第一个数据）* 第一个数据
normal_coff = (1-success_rate_conditioned_on_confidence[0]) * success_rate_conditioned_on_confidence[0]

success_rate_conditioned_on_confidence -= minus_data
# normal_coff = normal_coff * 0.5 * len(success_rate_conditioned_on_confidence)
success_rate_conditioned_on_confidence /= normal_coff
# assert normal_coff > 0
# 面积总和计算
sr_hr_area = {}
divider = 0.5 * (len(success_rate_conditioned_on_confidence) - 1)

sr_hr_area = np.trapz(success_rate_conditioned_on_confidence, dx=1) / divider

print('-'*15,'sh_hr_area','-'*15)
print(sr_hr_area)





# %%
