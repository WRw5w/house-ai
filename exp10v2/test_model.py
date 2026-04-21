import torch
import torch.nn as nn
from model_modified import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, base_features=32, p_drop=0.3).to(device)
x = torch.randn(1, 1, 128, 128).to(device)
y = model(x)
print(y.shape)
