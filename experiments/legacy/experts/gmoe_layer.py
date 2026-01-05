import torch
import torch.nn as nn
from .router import GeometricRouter

class BicameralGMoELayer(nn.Module):
    def __init__(self, dim, logic_expert, creative_expert, threshold=3.0):
        super().__init__()
        self.router = GeometricRouter(dim, threshold)
        self.logic_expert = logic_expert
        self.creative_expert = creative_expert

    def forward(self, x):
        is_logic, current_id = self.router(x)
        if is_logic:
            return self.logic_expert(x), "LOGIC", current_id
        else:
            return self.creative_expert(x), "CREATIVE", current_id
