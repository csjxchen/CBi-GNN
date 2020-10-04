import spconv
from torch import nn
# from ..utils import change_default_args, Sequential
from dets.ops.pointnet2 import pointnet2_utils
import torch
from dets.ops import pts_in_boxes3d
from dets.tools.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss
from dets.tools import tensor2points
import torch.nn.functional as F
from functools import partial
import models.necks.components as components
# from dets.ops.pointnet2.layers_utils import Grouper4, Grouper5, Grouper6
# from dets.
# from .bignn import BiGNN_V1, BiGNN_V2, BiGNN_V3
class CBiNet(nn.Module):
    def __init__(self, model_cfg):
        """The principal part of detector(CBi-GNN)
        Args:
            model_cfg (dict): define the architecture of model
        """
        self.model_cfg = model_cfg
        ThrDNet_cfg = self.model_cfg.ThrDNet
        _thrdnet_args = ThrDNet_cfg.copy()
        _thrdnet_type = _thrdnet_args.pop('type') 
        self.Thrdnet = components[_thrdnet_type](_thrdnet_args)

        TwoDNet_cfg = self.model_cfg.TwoDNet
        _twodnet_args = TwoDNet_cfg.copy()
        _twodnet_type = _twodnet_args.pop('type')
        self.Twodnet = components[_twodnet_type](_twodnet_args)
        
    def forward(self, data):
        # ! TODO
        pass