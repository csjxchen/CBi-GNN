
import torch.nn as nn

classs RPN(nn.Module):
    def __init(self, model_cfg, train_cfg, test_cfg):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        