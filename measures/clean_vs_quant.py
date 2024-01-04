import os
from transformers import DetrImageProcessor, ViTForImageClassification, AutoImageProcessor
import torch
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
import time
import pynvml

# Initialize NVML
pynvml.nvmlInit()


# Function to get total energy consumption
def get_total_energy_consumption(device_id=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)


imagenet = "/dt/shabtaia/dt-fujitsu/datasets/others/imagnet/ILSVRC/Data/DET/val"
Imglist = os.listdir(imagenet)

model_no_quant = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model_no_quant.to("cuda")
config_dict={"load_in_8bit": True, "load_in_4bit": False}
quant_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto",
                                                          load_in_8bit=True)
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
cuda_time_quant, cuda_mem_quant = [], []
cuda_time_no_quant, cuda_mem_no_quant = [], []
energy_consumption_quant, energy_consumption_no_quant = [], []

start = time.time()

# Warm-Up Inference

with torch.no_grad():
    dummy_input = torch.rand(1, 3, 224, 224).to("cuda")
    _ = quant_model(dummy_input)
    _ = model_no_quant(dummy_input)
    start_energy_quant = get_total_energy_consumption()
    pred_quant = quant_model(dummy_input).logits.sum().item()
    end_energy_quant = get_total_energy_consumption()
torch.cuda.synchronize()
num_images = 100
k = min(len(Imglist), num_images)
for img_dir in tqdm(Imglist[:k]):
    try:
        img = Image.open(os.path.join(imagenet, img_dir))
        inputs = image_processor(img, return_tensors="pt")["pixel_values"]
        inputs = inputs.to("cuda")

        temp_cuda_time_quant, temp_cuda_mem_quant = [], []
        temp_cuda_time_no_quant, temp_cuda_mem_no_quant = [], []
        for _ in range(1):
            # # Inference with non-quantized model
            # with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #                               profile_memory=True) as prof_no_quant:
            #     start_energy_quant = get_total_energy_consumption()
            #     pred_no_quant = model_no_quant(inputs).logits.sum().item()
            #     end_energy_quant = get_total_energy_consumption()
            # torch.cuda.synchronize()  # Ensure all CUDA operations are completed
            # energy_consumption_no_quant.append(end_energy_quant - start_energy_quant)
            # temp_cuda_time_no_quant.append(prof_no_quant.profiler.total_average().self_cuda_time_total / 1000.0)
            # temp_cuda_mem_no_quant.append(prof_no_quant.profiler.total_average().cuda_memory_usage / (2 ** 20))
            # # del inputs, pred_quant, pred_no_quant
            # torch.cuda.empty_cache()  # Clear unreleased memory

            # Inference with quantized model
            with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                          profile_memory=True) as prof_quant:
                start_energy_quant = get_total_energy_consumption()
                pred_quant = quant_model(inputs).logits.sum().item()
                end_energy_quant = get_total_energy_consumption()
            torch.cuda.synchronize()  # Ensure all CUDA operations are completed
            energy_consumption_quant.append(end_energy_quant - start_energy_quant)

            temp_cuda_time_quant.append(prof_quant.profiler.total_average().self_cuda_time_total / 1000.0)
            temp_cuda_mem_quant.append(prof_quant.profiler.total_average().cuda_memory_usage / (2 ** 20))

        cuda_time_quant.append(sum(temp_cuda_time_quant) / len(temp_cuda_time_quant))
        cuda_mem_quant.append(sum(temp_cuda_mem_quant) / len(temp_cuda_mem_quant))
        # cuda_time_no_quant.append(sum(temp_cuda_time_no_quant) / len(temp_cuda_time_no_quant))
        # cuda_mem_no_quant.append(sum(temp_cuda_mem_no_quant) / len(temp_cuda_mem_no_quant))
        avg_energy_consumption_quant = sum(energy_consumption_quant) / len(energy_consumption_quant)
        # avg_energy_consumption_no_quant = sum(energy_consumption_no_quant) / len(energy_consumption_no_quant)

    except Exception as e:
        print(f"Error processing {img_dir}: {e}")

# end = time.time()
# print("time: ", end - start)
avg_time_quant = sum(cuda_time_quant) / k
avg_mem_quant = sum(cuda_mem_quant) / k

# avg_time_no_quant = sum(cuda_time_no_quant) / k
# avg_mem_no_quant = sum(cuda_mem_no_quant) / k
# energy_consumption_no_quant = [i for i in energy_consumption_no_quant[1:] if i != 0]
energy_consumption_quant = [i for i in energy_consumption_quant[1:] if i != 0]
avg_energy_consumption_quant = sum(energy_consumption_quant) / len(energy_consumption_quant)
# avg_energy_consumption_no_quant = sum(energy_consumption_no_quant) / len(energy_consumption_no_quant)

print(f"Quant Model - Average GPU Time: {avg_time_quant} ms, Average GPU Memory: {avg_mem_quant} MB")
print(f"Quant Model footprint: {quant_model.get_memory_footprint()}")
print(f"Quant Model - Average Energy Consumption: {avg_energy_consumption_quant} mJ")

# print(f"Non-Quant Model - Average GPU Time: {avg_time_no_quant} ms, Average GPU Memory: {avg_mem_no_quant} MB")
# print(f"Non-Quant Model footprint: {model_no_quant.get_memory_footprint()}")
# print(f"Non-Quant Model - Average Energy Consumption: {avg_energy_consumption_no_quant} mJ")
