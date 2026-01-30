# Deep Additive Manufacturing

PyTorch implementations of deep learning models for predicting shape deformation in additive manufacturing (3D printing) processes.

## Overview

This repository contains neural network architectures based on the PredNet and U-Net designs for high-resolution shape deformation prediction. The models support both 2D and 3D convolutional operations for analyzing additive manufacturing data.

### Models

| Model | File | Description |
|-------|------|-------------|
| **PredNet (2D)** | `models/prednet.py` | 2D encoder-decoder network for 256x256 shape prediction |
| **PredNet (3D)** | `models/prednet3D.py` | 3D encoder-decoder network for 32x32x32 volumetric prediction |
| **PredNet Final** | `models/final.py` | Dual-branch architecture combining encoder and spatial branches for 16x16x16 output |
| **U-Net** | `models/unet.py` | Classic U-Net implementation for 2D segmentation |

## Installation

```bash
git clone https://github.com/lucky-verma/Deep-Additive-Manufacturing.git
cd Deep-Additive-Manufacturing
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from models.final import PredNet

model = PredNet()
input_tensor = torch.randn(1, 1, 16, 16, 16)
output_tensor = model(input_tensor)
print(output_tensor.shape)  # torch.Size([1, 16, 16, 16])
```

## Repository Structure

```
Deep-Additive-Manufacturing/
├── models/
│   ├── final.py        # Final PredNet with spatial branch (16x16x16)
│   ├── prednet.py      # 2D PredNet implementation (256x256)
│   ├── prednet3D.py    # 3D PredNet implementation (32x32x32)
│   └── unet.py         # U-Net implementation
├── papers/             # Reference research papers
├── models.ipynb        # Jupyter notebook with model experiments
├── LICENSE
└── README.md
```

## Tech Stack

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch
- **Notebook**: Jupyter

## References

This implementation is based on:
- Zhen Shen et al., "PredNet and CompNet: Prediction and High-Precision Compensation of In-Plane Shape Deformation for Additive Manufacturing"
- "High-Resolution Shape Deformation Prediction in Additive Manufacturing Using 3D CNN"

## License

MIT License
