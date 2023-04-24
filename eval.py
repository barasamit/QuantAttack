import pandas as pd
# from datasets import load_dataset
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from dataloader.general_loader import *
from utils.init_collect_arrays import outliers_arr
from torch.profiler import profile, record_function, ProfilerActivity


def calc_model_time(x):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    rgb = model(x)
    end.record()
    torch.cuda.synchronize()
    total = start.elapsed_time(end)

    outliers_arr.clear()

    return total / 1000.0  # convert time to seconds


def to_pandas_df(prof):
    results = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10)
    columns = [s for s in results.split("\n")[1].strip().split("  ") if
               s != results.split("\n")[1].strip().split("  ")[1]]
    data = []
    for i in range(3, 12):
        data.append([s for s in results.split("\n")[i].strip().split("  ") if
                     s != results.split("\n")[i].strip().split("  ")[1]])
    df = pd.DataFrame(data, columns=columns)
    return df


device = torch.device("cuda")

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto", load_in_8bit=True)
model.eval()
img_before = "/sise/home/barasa/8_bit/images_changes/0.jpg"
img_after = "/sise/home/barasa/8_bit/images_changes/98.jpg"
inputs = feature_extractor(images=Image.open(img_before), return_tensors="pt")
inputs2 = feature_extractor(images=Image.open(img_after), return_tensors="pt")

inputs_pic = inputs["pixel_values"]
with profile(activities=[
    ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs_pic)
results = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
print(results)
df = to_pandas_df(prof)
print(df[["Name","CUDA Mem"]])









# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))


# clean_time = [calc_model_time(inputs_pic) for _ in range(150)]
# print("mean time: ", mean(clean_time[1:]), "std: ", np.std(clean_time[1:]))


# Define the transforms to be applied to the validation/test data
# val_transforms = transforms.Compose([
#
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# device = torch.device("cuda")
#
# # val_dataset = datasets.ImageFolder("/dt/shabtaia/dt-fujitsu/datasets/active/pascal_voc/images/val", transform=val_transforms)
# val_dataloader = GeneralLoader("/dt/shabtaia/dt-fujitsu/datasets/active/imagenet_detection/")
# num_classes = val_dataloader.n_classes
#
# model.eval()
# # model.to(device)
# y_true = []
# y_pred_real = []
# y_pred_8bit = []
#
#
# img_before = "/sise/home/barasa/8_bit/images_changes/0.jpg"
# img_after = "/sise/home/barasa/8_bit/images_changes/98.jpg"
# inputs1 = feature_extractor(images=Image.open(img_before), return_tensors="pt")
# inputs2 = feature_extractor(images=Image.open(img_after), return_tensors="pt")
# outputs = model(inputs1["pixel_values"].to(device))
# outputs2 = model(inputs2["pixel_values"].to(device))
# logits1 = outputs.logits
# logits2 = outputs2.logits
# predicted_class_idx1 = logits1.argmax(-1).item()# model predicts one of the 1000 ImageNet classes
# predicted_class_idx2 = logits2.argmax(-1).item()# model predicts one of the 1000 ImageNet classes
# print("Predicted class:", model.config.id2label[predicted_class_idx1])
# print("Predicted class:", model.config.id2label[predicted_class_idx2])


# # with torch.no_grad():
#     for images, labels in tqdm(val_dataloader):
#
#         try:
#             inputs = feature_extractor(images=images, return_tensors="pt")
#         except:
#             continue
#
#         outputs = model(inputs["pixel_values"])
#         outputs2 = model2(inputs["pixel_values"])
#         logits1 = outputs.logits
#         logits2 = outputs2.logits
#         predicted_class_idx1 = logits1.argmax(-1).item()# model predicts one of the 1000 ImageNet classes
#         predicted_class_idx2 = logits2.argmax(-1).item()# model predicts one of the 1000 ImageNet classes
#         y_real.append(predicted_class_idx1)
#         y_8bit.append(predicted_class_idx2)
#         # y_true.append(labels)
#         # y_pred.append(predicted_class_idx)
#         if predicted_class_idx1 != predicted_class_idx2:
#             print("Predicted class:", model.config.id2label[predicted_class_idx1])
#
# # Compute the accuracy score for the model predictions
# accuracy = accuracy_score(y_true, y_pred)
# print('Accuracy:', accuracy)
# with open("/sise/home/barasa/8_bit/image_net_map.json", 'r') as file:
#     map_dict = json.load(file)
#
# root = "/dt/shabtaia/dt-fujitsu-availability/datasets/imagenet/ILSVRC2012/val/"
# clss_list = os.listdir(root)
# output_dir = "/sise/home/barasa/8_bit/output/"
#
# for cls in tqdm(clss_list):
#     clss_list = os.listdir(os.path.join(root, cls))
#
#     for i, d in enumerate((clss_list)):
#         img = Image.open(os.path.join(root, cls)+"/"+ d)
#         try:
#             inputs = feature_extractor(images=img, return_tensors="pt")
#         except:
#             continue
#         with torch.no_grad():
#             outputs = model(inputs["pixel_values"].to(device))
#             outputs2 = model2(inputs["pixel_values"].to(device))
#         predicted_class_idx1 = outputs.logits.argmax(-1).item()  # model predicts one of the 1000 ImageNet classes
#         predicted_class_idx2 = outputs2.logits.argmax(-1).item()# model predicts one of the 1000 ImageNet classes
#         y_pred_real.append(map_dict[str(predicted_class_idx1)])
#         y_pred_8bit.append(map_dict[str(predicted_class_idx2)])
#         y_true.append(cls)
#
#
# num_matches1 = sum(1 for i, j in zip(y_pred_real, y_true) if i == j)
# num_matches2 = sum(1 for i, j in zip(y_pred_8bit, y_true) if i == j)
#
# # Calculate the accuracy as a percentage
# accuracy1 = (num_matches1 / len(y_pred_real)) * 100
# accuracy2 = (num_matches2 / len(y_pred_real)) * 100
# # Print the accuracy
# print(f"Accuracy with regular: {accuracy1}%")
# print(f"Accuracy with 8_bit: {accuracy2}%")
#
#
#
#
# print("done")
