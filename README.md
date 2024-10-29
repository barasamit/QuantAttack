# QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers

[![Published at WACV 2025](https://img.shields.io/badge/Published-WACV%202025-blue.svg)](https://arxiv.org/pdf/2312.02220)
[![arXiv](https://img.shields.io/badge/arXiv-2312.02220-b31b1b.svg)](https://arxiv.org/abs/2312.02220)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of **QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers** (WACV 2025) by Amit Baras, Alon Zolfi, Asaf Shabtai, and Yuval Elovici.

## Overview

QuantAttack is a novel attack method that targets the availability of quantized vision transformers. By exploiting the dynamic behavior of quantization techniques, our attack can significantly impact model efficiency by:
- Slowing down inference time
- Increasing memory usage
- Raising energy consumption

### Abstract

In recent years, there has been a significant trend in deep neural networks (DNNs), particularly transformer-based models, of developing ever-larger and more capable models. While they demonstrate state-of-the-art performance, their growing scale requires increased computational resources (e.g., GPUs with greater memory capacity). 

To address this problem, quantization techniques (i.e., low-bit-precision representation and matrix multiplication) have been proposed. Most quantization techniques employ a static strategy in which the model parameters are quantized, either during training or inference, without considering the test-time sample. 

In contrast, dynamic quantization techniques, which have become increasingly popular, adapt during inference based on the input provided while maintaining full-precision performance. However, their dynamic behavior and average-case performance assumption make them vulnerable to a novel threat vector – adversarial attacks that target the model's efficiency and availability.


## Getting Started

### Prerequisites

- CUDA-capable GPU
- Python 3.8+
- PyTorch

### Installation

> **Note:** Follow these commands in the specified order for a successful installation.

```bash
# Create and activate conda environment
conda create -n QuantAttack python=3.8
conda activate QuantAttack

# Install CUDA toolkit
conda install cudatoolkit

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install transformers accelerate datasets bitsandbytes pandas fvcore
```

## Dataset Preparation

Organize your datasets (e.g., ImageNet or COCO) following this structure:
```
dataset_name/
├── images/
│   ├── train/
│   │   ├── file1.png
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
```

## Usage

### Configuration

Model and attack parameters can be configured in the [config file](https://github.com/barasamit/8_bits_attack/blob/main/configs/attacks_config.py).

### Running Attacks

1. Single Attack:
```bash
python many_to_many_attack.py
```

2. Universal Attack:
```bash
python universal_attack.py
```

### Evaluation

To evaluate attack performance:
```bash
python single_eval_only.py
```

> **Note:** Specify the path to adversarial images in the main function.

## Results

[Add key results, graphs, or performance metrics here]

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{baras2023quantattackexploitingdynamicquantization,
    title={QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers}, 
    author={Amit Baras and Alon Zolfi and Yuval Elovici and Asaf Shabtai},
    year={2023},
    eprint={2312.02220},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2312.02220}
}
```


