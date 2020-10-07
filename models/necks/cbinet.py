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
from models.necks.components import  threed_models, twod_models
# from dets.ops.pointnet2.layers_utils import Grouper4, Grouper5, Grouper6
# from dets.
# from .bignn import BiGNN_V1, BiGNN_V2, BiGNN_V3
class CBiNet(nn.Module):
    def __init__(self, model_cfg):
        
        """The principal part of detector(CBi-GNN)
        Args:
            model_cfg (dict): define the architecture of model
        """
        super(CBiNet, self).__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = self.model_cfg['output_shape']
        ThrDNet_cfg = self.model_cfg.ThrDNet
        # _thrdnet_args = ThrDNet_cfg.copy()
        # _thrdnet_type = _thrdnet_args.pop('type') 
        
        self.Thrdnet = threed_models[ThrDNet_cfg.type](ThrDNet_cfg.args)
        TwoDNet_cfg = self.model_cfg.TwoDNet
        # _twodnet_args = TwoDNet_cfg.copy()
        # _twodnet_type = _twodnet_args.pop('type')
        self.Twodnet = twod_models[TwoDNet_cfg.type](TwoDNet_cfg.args)
        
    def forward(self, data):
        """
        Args:
            data (dict):  {"voxel_input", "coords",  "batch_size"}
        Return:
            bev_features: for rpn head
            alignment_features: for alignment head
        """
        x = spconv.SparseConvTensor(data['voxel_input'], data['coords'], self.sparse_shape, data['batch_size'])
        # ! TODO
        x = self.Thrdnet(x, batch_size=data['batch_size'])
        x = x.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)
        rpn_head_features, align_head_features = self.Twodnet(x)
        outs = {
            "rpn_head_features": rpn_head_features, 
            "align_head_features": align_head_features
        }
        return outs