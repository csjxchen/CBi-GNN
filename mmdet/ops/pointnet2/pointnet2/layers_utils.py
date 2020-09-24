# import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pointnet2_utils
from . import pytorch_utils as pt_utils
from typing import List
import torch
class Grouper(nn.Module):
    def __init__(self, radius, nsamples, mlps, use_xyz=True, bn=True, instance_norm=False, pool_method='max_pool'):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.pool_method = pool_method
        self.radius = radius[0]
        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features) # (B, C, npoint, nsample)
            # print('new_features', new_features.shape)
            # print('xyz', xyz.shape)
            # print('new_xyz', new_xyz.transpose(1, 2).unsqueeze(-1).shape)


            # print((torch.abs(new_features[:, :3, :, :] - new_xyz.transpose(1, 2).unsqueeze(-1))<self.radius).float().mean())
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)
        
        return new_xyz, torch.cat(new_features_list, dim=1)


class Grouper_2(nn.Module):
    def __init__(self, radius, nsamples, mlps, use_xyz=True, bn=True, instance_norm=False, pool_method='max_pool'):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.pool_method = pool_method

        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features) # (B, C, npoint, nsample)
            
            
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)


        
        return new_xyz, torch.cat(new_features_list, dim=1)

class Grouper3(nn.Module):
    def __init__(self, radius, nsamples, mlps, use_xyz=True, bn=True, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.pool_methods = pool_methods
        self.radius = radius[0]
        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            mlp_spec = mlps[i]
            
            if use_xyz:
                mlp_spec[0] += 3
            
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)         # (B, C, npoint, nsample)

            
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            pooled_features = []
            for pool_method in self.pool_methods:       
                if pool_method == 'max_pool':
                    pooled_feature = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                    pooled_features.append(pooled_feature)
                elif pool_method == 'avg_pool':
                    pooled_feature = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                    pooled_features.append(pooled_feature)

                else:
                    raise NotImplementedError
            new_features = torch.cat(pooled_features, dim=1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)
        
        return new_xyz, torch.cat(new_features_list, dim=1)


class Grouper4(nn.Module):
    def __init__(self, radius, nsamples, mlps, use_xyz=True, bn=True, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.pool_methods = pool_methods
        self.radius = radius[0]
        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            mlp_spec = mlps[i]
            
            if use_xyz:
                mlp_spec[0] += 3
            
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)         # (B, C, npoint, nsample)
            new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
            dist = torch.norm(new_features_xyz, dim=1, keepdim=True)        # (B, 1, npoints, nsample)
            dist_recip = 1.0 / (dist + 1e-8)                                # (B, 1, npoints, nsample)
            norm = torch.sum(dist_recip, dim=3, keepdim=True)               # (B, 1, npoints, 1)
            weights = dist_recip / norm                                      # (B, 1, npoints, nsamples)
            # new_features[:, 3:, :, :] = weight * new_features[:, 3:, :, :]  # (B, 1, npoints, 1)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            # print(weights.shape, norm.shape, dist.shape, new_features.shape)
            new_features = weights * new_features
            new_features = new_features.sum(dim=3, keepdim=True)
            #     #     pooled_feature = F.max_pool2d(
            #     #         new_features, kernel_size=[1, new_features.size(3)]
            #     #     )  # (B, mlp[-1], npoint, 1)
            #     #     pooled_features.append(pooled_feature)
            #     # elif pool_method == 'avg_pool':
            #     #     pooled_feature = F.avg_pool2d(
            #     #         new_features, kernel_size=[1, new_features.size(3)]
            #     #     )  # (B, mlp[-1], npoint, 1)
            #     #     pooled_features.append(pooled_feature)

            #     # else:
            #     #     raise NotImplementedError
            # new_features = torch.cat(pooled_features, dim=1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)
        
        return new_xyz, torch.cat(new_features_list, dim=1)
