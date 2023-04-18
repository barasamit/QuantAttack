#
#
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import torch
# import torch.nn as nn
#
# import bitsandbytes as bnb
# from bitsandbytes.nn import Linear8bitLt
# model_name = "t5-3b-sharded"
# model_id=f"ybelkada/{model_name}"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model_8bit = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto", load_in_8bit=False)
# #
# #
# # max_new_tokens = 50
# #
# # input_ids = tokenizer(
# #     "translate English to German: Hello my name is Younes and I am a Machine Learning Engineer at Hugging Face", return_tensors="pt"
# # ).input_ids
# #
# # outputs = model_8bit.generate(input_ids, max_new_tokens=max_new_tokens)
# # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# #
#
# fp16_model = nn.Sequential(
#     nn.Linear(64, 64),
#     nn.Linear(64, 64)
# )
#
# int8_model = nn.Sequential(
#     Linear8bitLt(64, 64, has_fp16_weights=False),
#     Linear8bitLt(64, 64, has_fp16_weights=False)
# )
#
# int8_model.load_state_dict(model_8bit)
# int8_model = int8_model.to(0) # Quantization happens here

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
from torch.profiler import profile, record_function, ProfilerActivity

name = "bigscience/bloom-3b"
text = "Hello my name is"
max_new_tokens = 20

count = 0

def generate_from_model(model, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt')
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
    print(count)
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)


model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)
print(generate_from_model(model_8bit, tokenizer))




# # model_native = AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype="auto")
# #
# #
# #
# # mem_fp16 = model_native.get_memory_footprint()
# # mem_int8 = model_8bit.get_memory_footprint()
#
# generate_from_model(model_8bit, tokenizer)
# print("Memory footprint int8 model: {} | Memory footprint fp16 model: {} | Relative difference: {}".format(mem_int8,
#                                                                                                            mem_fp16,
#                                                                                                            mem_fp16 / mem_int8))
