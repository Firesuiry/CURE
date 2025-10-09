from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor
from transformers import LogitsProcessorList, LogitsProcessor
import torch

class LLAMA:
    def __init__(self, model_name,load_in_8bit=False, cache_dir=None):
        self.device = torch.device("cuda")
        # torch.cuda.set_per_process_memory_fraction(0.8, 0)  # GPU 0
        # torch.cuda.set_per_process_memory_fraction(0.8, 1)  # GPU 1
        # torch.cuda.set_per_process_memory_fraction(0.85, 2)  # GPU 2
        # torch.cuda.set_per_process_memory_fraction(0.75, 3)  # GPU 3
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

if __name__ == '__main__':
    
    prompt = '''
    Please list the canned beverages listed below:
'a Coke'
'a Pepsi'
'a RedBull'
'a Sprite'
'a bottled unsweetened tea'
'a bottled water'
'a orange soda'
'a Fanta'
'a Dr Pepper'
'a Mountain Dew'
'a Ginger ale'
'a Diet Coke'
'a Diet Pepsi'
'a energy drink'
'a root beer'
'a lemon-lime soda'
'a fruit punch'
'a sparkling juice'
'a iced coffee'
'a milk'
'a juice box'
'a sports drink'
'a kombucha'
'a kefir'
'a flavored seltzer'
'a flavored sparkling water'
'a coconut water'
'a aloe vera drink'
'a ginseng drink'
'a yerba mate'
'a green tea'
'a black tea'
'a herbal tea'
'a fruit infused water'
'a flavored milk'
'a eggnog'
'a hot chocolate'
'a cold brew'
'a matcha latte'
'a chai latte'
'a frappuccino'
'''
    llama = LLAMA("voidful/Llama-3.2-8B-Instruct", load_in_8bit=True)
    # a,b = llama.llama(prompt, stop_seq=["\n\n"])
    # print(a)
    # print(prompt)
    # print(b)
    # for i in range(10):
    a,b = llama.llama(prompt,max_length=8000)
    print(b)

    # model_id = "voidful/Llama-3.2-8B-Instruct"
    # model_id = "unsloth/Llama-3.3-70B-Instruct"
    # device = torch.device("cuda")
    # model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto",cache_dir='/data1/yinshiyuan/')
    # tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir='/data1/yinshiyuan/')
    # messages = [
    #     {"role": "system", "content": "你是一个机器人，根据人类指示执行命令。你现在在厨房，面前有桃子，雪碧和米饭"},
    #     {"role": "user", "content": "给我点喝的"},
    #     {"role": "system", "content": "请一步一步思考你需要怎么做"},
    # ]

    # input_ids = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     return_tensors="pt"
    # ).to(model.device)
    # print(input_ids)
    # decode_outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    # print(decode_outputs[0])
    # terminators = [
    # tokenizer.eos_token_id,
    # tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]

    # outputs = model.generate(
    #     input_ids,
    #     max_new_tokens=256,
    #     eos_token_id=terminators,
    #     do_sample=True,
    #     temperature=0.6,
    #     top_p=0.9,
    # )
    # response = outputs[0][input_ids.shape[-1]:]
    # print(tokenizer.decode(response, skip_special_tokens=True))
    # messages.append({"role": "assistant", "content": tokenizer.decode(response, skip_special_tokens=True)})
    # messages.append({"role": "system", "content": '根据上面的思考，你需要怎么做？并在答案末尾给出你对你做法的信心，格式为百分数'})
    # input_ids = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     return_tensors="pt"
    # ).to(model.device)
    # print(input_ids)
    # decode_outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    # print(decode_outputs[0])
    # outputs = model.generate(
    #     input_ids,
    #     max_new_tokens=256,
    #     eos_token_id=terminators,
    #     do_sample=True,
    #     temperature=0.6,
    #     top_p=0.9,
    # )
    # response = outputs[0][input_ids.shape[-1]:]
    # print(tokenizer.decode(response, skip_special_tokens=True))