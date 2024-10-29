# QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers
[![Published at WACV 2025](https://img.shields.io/badge/Published-WACV%202025-blue.svg)](link_to_paper)

This is a PyTorch implementation of [QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers](https://arxiv.org/pdf/2312.02220) by Amit Baras, Alon Zolfi, Asaf Shabtai, Yuval Elovici.

## Abstract
Quantization techniques are essential for reducing computational and memory demands in transformer models, enabling deployment on edge devices. Dynamic quantization has become increasingly popular due to its ability to adapt precision based on input data, preserving model performance at low resource costs. However, this adaptive behavior exposes models to a new attack vector. We present *QuantAttack*, an adversarial attack targeting the availability of vision transformers using dynamic quantization, designed to degrade performance through increased memory, processing time, and energy consumption. QuantAttack creates adversarial examples that trigger high-bit operations, thereby slowing down the model’s inference. Our evaluation demonstrates significant increases in memory usage, processing time, and energy consumption in ViT and DeiT models. We also propose several countermeasures to mitigate the effects of QuantAttack.

add image here

## Installation - the order is important
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
The datasets (imagnet/coco) should be formatted in the following format:
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



## Citation
```

}
```

