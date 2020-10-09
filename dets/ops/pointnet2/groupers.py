import torch.nn as nn
import torch.nn.functional as F
from . import pointnet2_utils
from . import pytorch_utils as pt_utils
from typing import List
import torch

class GrouperDisAttention(nn.Module):
    def __init__(self, 
                radius, 
                nsamples, 
                mlps, 
                use_xyz=True, 
                bn=False, 
                instance_norm=False, 
                pool_methods=['max_pool']):
        super(GrouperDisAttention, self).__init__()
        # self.groupers = nn.ModuleList()
        # self.mlps = nn.ModuleList()
        # self.xyz_mlps = nn.ModuleList()
        xyz_mlp_spec = [3, 32]
        mlp_spec = mlps
        self.pool_methods = pool_methods
        self.radius = radius
        self.nsamples = nsamples
        self.groupers = pointnet2_utils.QueryAndGroup(self.radius, self.nsamples, use_xyz=use_xyz)            
        # mlp_spec = mlps[i]
        # xyz_mlp_spec = xyz_mlps[i]
        self.mlps = pt_utils.SharedMLP(mlp_spec, bn=False, instance_norm=instance_norm)
        self.xyz_mlps = pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=instance_norm)    
    
    
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        new_features, idn = self.groupers(xyz, new_xyz, features)  # (B, C, npoint, nsample)
        # idn(B, npoints, nsamples)
        # (B, npoints)        
        idn_sum = torch.sum(idn, dim=2)
        # calculate which neighboring query point alone
        # B, npoints, 1
        idn_mask = (idn_sum > 0).unsqueeze(-1).float()
        
        # print(f"idn_mask, {idn_mask.shape}")
        
        # idn_sum = torch.sum(idn, dim=2, keepdim=True)
        
        idn_index = torch.nonzero(idn_mask)  
        
        _idn = torch.cuda.IntTensor(idn.shape[0], idn.shape[1], idn.shape[2]).zero_() + 1
        
        # copy the valid id repeat number to _idn, unvalid are for 1 
        _idn[idn_index[:, 0], idn_index[:, 1], :] = idn[idn_index[:, 0], idn_index[:, 1], :]
        
        _idn = _idn.unsqueeze(1).float() # B, 1, npoints, nsamples
        
        new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
        
        new_features = new_features[:, 3:, :, :]
        
        dist = torch.norm(new_features_xyz, dim=1, keepdim=True)        # (B, 1, npoints, nsample)
        
        dist_recip = 1.0 / (dist + 1e-8)                                # (B, 1, npoints, nsample)
                    
        # dist_recip = 1.0 / (dist + 1e-8)                              # (B, 1, npoints, nsample)
        
        dist_recip /= _idn                                              # denominator
        
        # print(dist_recip.squeeze(1)[0, 0], _idn.squeeze(1)[0, 0])
        
        norm = torch.sum(dist_recip, dim=3, keepdim=True)               # (B, 1, npoints, 1)
        
        weights = dist_recip / norm                                     # (B, 1, npoints, nsamples)

        weights = weights * idn_mask.unsqueeze(1)                       # detach the query point with no neighbouring points
        
        # new_features[:, 3:, :, :] = weight * new_features[:, 3:, :, :]  # (B, 1, npoints, 1)
        
        new_features = self.mlps(new_features)  # (B, mlp[-1], npoint, nsample)
        
        new_features = weights * new_features

        new_features_xyz = self.xyz_mlps(new_features_xyz)
        
        new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
        
        new_features = new_features.sum(dim=3, keepdim=True)
        
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        
        new_features_xyz = new_features_xyz.squeeze(-1)
        
        new_features  = torch.cat([new_features_xyz, new_features], dim=1)
        
        # new_features_list.append(new_features)
        
        return new_xyz, new_features