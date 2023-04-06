# PredNet PyTorch Implementation

### Model Zoo

This repository contains a collection of models implemented in PyTorch. The models are organized by category and are accompanied by a short description and example code.

Here's a PyTorch implementation of PredNet to perform 3D convolutional prediction. The code is based on the original U-Net architecture. 


```python
import torch
from models.final import PredNet

model = PredNet()
input_tensor = torch.randn(1, 1, 32, 32, 32)
output_tensor = model(input_tensor)
```