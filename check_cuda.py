from transformers import DetrImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset


dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = DetrImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", device_map="auto",load_in_8bit=True)


inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])