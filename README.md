# QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers
[![Published at WACV 2025](https://img.shields.io/badge/Published-WACV%202025-blue.svg)](https://arxiv.org/pdf/2312.02220)

This repository contains a PyTorch implementation of **QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers** by Amit Baras, Alon Zolfi, Asaf Shabtai, and Yuval Elovici.


### Abstract
In recent years, there has been a significant trend in deep neural networks (DNNs), particularly transformer-based models, of developing ever-larger and more capable models. While they demonstrate state-of-the-art performance, their growing scale requires increased computational resources (e.g., GPUs with greater memory capacity). To address this problem, quantization techniques (i.e., low-bit-precision representation and matrix multiplication) have been proposed. Most quantization techniques employ a static strategy in which the model parameters are quantized, either during training or inference, without considering the test-time sample. In contrast, dynamic quantization techniques, which have become increasingly popular, adapt during inference based on the input provided while maintaining full-precision performance. However, their dynamic behavior and average-case performance assumption make them vulnerable to a novel threat vector – adversarial attacks that target the model’s efficiency and availability. In this paper, we present QuantAttack, a novel attack that targets the availability of quantized vision transformers, slowing down the inference and increasing memory usage and energy consumption.

![Insert your image here](link_to_image)

## Installation
> **Note:** The order of commands is important for a successful installation.
```bash
conda create -n QuantAttack python=3.8
conda activate QuantAttack
conda install cudatoolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install accelerate
pip install datasets
pip install bitsandbytes
pip install pandas
pip install fvcore
```

## Datasets
Prepare your datasets (e.g., ImageNet or COCO) with the following structure:

```
├── dataset_name
│   ├── images
│   │   ├── train
│   │   │   ├── file1.png
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── ...

```

## Usage
### Configuration

Configurations can be changed in the [config](https://github.com/barasamit/8_bits_attack/blob/main/configs/attacks_config.py) file.

### Train perturbation

Run the [many_to_many_attack.py](https://github.com/barasamit/8_bits_attack/blob/main/many_to_many_attack.py) file for single attack.
or 
[many_to_many_attack.py](https://github.com/barasamit/8_bits_attack/blob/main/universal_attack.py) file for universal attack.

### Test

Run the [single_eval_only.py](https://github.com/barasamit/8_bits_attack/blob/main/single_eval_only.py) file. Specify the location of the adversarial images in main function.

## Citation
```
@misc{baras2023quantattackexploitingdynamicquantization,
      title={QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers}, 
      author={Amit Baras and Alon Zolfi and Yuval Elovici and Asaf Shabtai},
      year={2023},
      eprint={2312.02220},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2312.02220}, 
}
}
```

