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
from dets.ops.pointnet2.layers_utils import Grouper4, Grouper5, Grouper6
# from .bignn import BiGNN_V1, BiGNN_V2, BiGNN_V3
class CBiNet(nn.Module):
    def __init__(self, model_cfg):
        self


