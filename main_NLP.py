import torch
from transformers import AutoTokenizer, BloomForSequenceClassification
from utils.init_collect_arrays import input_arr, outliers_arr, outliers_arr_local, pointers

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-3b",device_map="auto", load_in_8bit=True)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits


print(logits)
