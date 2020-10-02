import torch.nn as nn 

class Detector(nn.Module):
    def __init__(self, model_cfg,
                train_cfg, 
                test_cfg
                ):
        """Container for generalized detector
        Args:
            model_cfg (dict): definite architecture of models
            train_cfg (dict): setting in train_forward
            test_cfg (dict): [description]
        """
        if "rpn" in model_cfg.keys():
            self.rpn  = RPN(model_cfg.rpn)
        
        if "rfn" in model_cfg.keys():
            self.rfn  = RPN(model_cfg.rfn)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
