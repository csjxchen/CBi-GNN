import torch.nn as nn

class BEVNet(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def forward(self, data):
        