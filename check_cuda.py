import os
from transformers import DetrImageProcessor, ViTForImageClassification,AutoImageProcessor
import torch
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

imagenet = "/dt/shabtaia/dt-fujitsu/datasets/others/imagnet/ILSVRC/Data/DET/val"
Imglist = os.listdir(imagenet)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto",
                                                          load_in_8bit=True)
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
for j in range(10):
    for i in tqdm(range(50)):
        try:
            img_dir = os.path.join(imagenet, Imglist[i])
            img = Image.open(img_dir)
            # calac time for each image
            inputs = image_processor(img, return_tensors="pt")["pixel_values"]
            # inputs = torch.rand(1, 3, 224, 224)
            pred = model(inputs)
            del inputs, pred
            torch.cuda.empty_cache()
        except:
            continue

