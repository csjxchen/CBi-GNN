import torch
from torch import nn
from torch.nn import functional as F
# from ..utils import change_default_args, Empty, get_paddings_indicator

class SimpleVoxel(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(SimpleVoxel, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
    
    def forward(self, features, num_voxels):
        return features
    