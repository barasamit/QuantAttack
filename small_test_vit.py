from torch import nn
# from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import torch



# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# img = "/dt/shabtaia/dt-fujitsu/datasets/others/imagnet/ILSVRC/Data/DET/val/ILSVRC2012_val_00000001.JPEG"
# image = Image.open(img)
# # image = Image.open(requests.get(url, stream=True).raw)
#
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto", load_in_8bit=True)
#
# inputs = feature_extractor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# #
# list1 = torch.stack([tensor for tensor in input_arr if tensor.size() == (1, 197, 768)])
# list2 = torch.stack([tensor for tensor in input_arr if tensor.size() == (1, 197, 3072)])
#
# list1_max = list1.max(dim=3)[0].squeeze()
# list2_max = list2.max(dim=3)[0].squeeze()
#
# target1 = torch.full((60,197), 7).to('cuda')
# target2 = torch.full((12,197), 7).to('cuda')
#
# loss = nn.MSELoss()
#
# output1 = loss(list1_max, target1)
# output2 = loss(list2_max, target2)
#
#
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print(predicted_class_idx)
# print("Predicted class:", model.config.id2label[predicted_class_idx])
