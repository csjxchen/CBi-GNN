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
    def __init__(self, radius, nsamples, mlps, use_xyz=True, bn=False, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        self.xyz_mlps = nn.ModuleList()
        xyz_mlps = [[3, 32]]
        self.pool_methods = pool_methods
        self.radius = radius[0]
        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            mlp_spec = mlps[i]
            xyz_mlp_spec = xyz_mlps[i]
            
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=False, instance_norm=instance_norm))
            self.xyz_mlps.append(pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=instance_norm))
            
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features, idn = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
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
            
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            new_features = weights * new_features

            new_features_xyz = self.xyz_mlps[i](new_features_xyz)
            
            new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
            
            new_features = new_features.sum(dim=3, keepdim=True)
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            
            new_features_xyz = new_features_xyz.squeeze(-1)
            
            new_features  = torch.cat([new_features_xyz, new_features], dim=1)
            
            new_features_list.append(new_features)
            
        return new_xyz, torch.cat(new_features_list, dim=1)


class Grouper5(nn.Module):
    def __init__(self, radius, nsamples, mlps, query_ch, neigh_ch, use_xyz=True, bn=False, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
         
        self.xyz_mlps = nn.ModuleList()
        self.query_feats_mlps  =nn.ModuleList()
        self.ad_xyz_mlps  =nn.ModuleList()
        self.ad_feats_mlps  =nn.ModuleList()

        xyz_mlps = [[3, 16]]
        self.pool_methods = pool_methods
        self.radius = radius[0]

        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            mlp_spec = mlps[i] + [neigh_ch]
            xyz_mlp_spec = xyz_mlps[i]
            
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=False, instance_norm=instance_norm))
            self.query_feats_mlps.append(pt_utils.SharedMLP([query_ch, neigh_ch], bn=False, instance_norm=instance_norm))
            self.ad_xyz_mlps.append(pt_utils.SharedMLP([3, neigh_ch], bn=False, instance_norm=instance_norm))
            self.ad_feats_mlps.append(pt_utils.SharedMLP([neigh_ch, neigh_ch], bn=False, instance_norm=instance_norm))
            self.xyz_mlps.append(pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=instance_norm))
            
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None, query_features: torch.Tensor=None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features, idn = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            # idn(B, npoints, nsamples)
            # (B, npoints)
            
            idn_sum = torch.sum(idn, dim=2)
            
            idn_mask = (idn_sum > 0).unsqueeze(-1).float()
            
            # print(f"idn_mask, {idn_mask.shape}")
            
            # idn_sum = torch.sum(idn, dim=2, keepdim=True)
            # print(query_features.shape)
            query_new_features = query_features.unsqueeze(-1)
            
            query_new_features = self.query_feats_mlps[i](query_new_features)
            

            idn_index = torch.nonzero(idn_mask)  
            
            _idn = torch.cuda.IntTensor(idn.shape[0], idn.shape[1], idn.shape[2]).zero_() + 1
            
            _idn[idn_index[:, 0], idn_index[:, 1], :] = idn[idn_index[:, 0], idn_index[:, 1], :]
            
            _idn = _idn.unsqueeze(1).float()
            
            new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
            
            new_features = new_features[:, 3:, :, :]
            
            feats_dist = query_new_features - new_features

            feats_dist = self.ad_feats_mlps[i](feats_dist)
            
            space_dist = self.ad_xyz_mlps[i](new_features_xyz)

            bi_edge = feats_dist * space_dist  # (B, ch, npoints, nsample)

            dist = torch.norm(new_features_xyz, dim=1, keepdim=True)        # (B, 1, npoints, nsample)
            
            dist_recip = 1.0 / (dist + 1e-8)                                # (B, 1, npoints, nsample)
            
            # dist_recip = 1.0 / (dist + 1e-8)                              # (B, 1, npoints, nsample)
            
            dist_recip /= _idn
            
            # print(dist_recip.squeeze(1)[0, 0], _idn.squeeze(1)[0, 0])
            
            norm = torch.sum(dist_recip, dim=3, keepdim=True)               # (B, 1, npoints, 1)
            
            weights = dist_recip / norm                                     # (B, 1, npoints, nsamples)

            weights = weights * idn_mask.unsqueeze(1)
            
            # new_features[:, 3:, :, :] = weight * new_features[:, 3:, :, :]  # (B, 1, npoints, 1)
                       
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            new_features = weights * new_features 
            
            new_features = new_features * bi_edge
            
            new_features_xyz = self.xyz_mlps[i](new_features_xyz)
            
            new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
            
            new_features = new_features.sum(dim=3, keepdim=True)
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            
            new_features_xyz = new_features_xyz.squeeze(-1)
            
            new_features  = torch.cat([new_features_xyz, new_features], dim=1)
            
            new_features_list.append(new_features)
            
        return new_xyz, torch.cat(new_features_list, dim=1)

class Grouper6(nn.Module):
    def __init__(self, radius, nsamples, mlps, query_ch, neigh_ch, use_xyz=True, bn=False, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
         
        self.xyz_mlps = nn.ModuleList()
        self.query_feats_mlps  =nn.ModuleList()
        self.ad_xyz_mlps  =nn.ModuleList()
        self.ad_feats_mlps  =nn.ModuleList()

        xyz_mlps = [[3, 32]]
        self.pool_methods = pool_methods
        self.radius = radius[0]

        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            mlp_spec = mlps[i]
            xyz_mlp_spec = xyz_mlps[i]
            
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=False, instance_norm=instance_norm))
            self.query_feats_mlps.append(pt_utils.SharedMLP([query_ch, neigh_ch], bn=False, instance_norm=instance_norm))
            self.ad_xyz_mlps.append(pt_utils.SharedMLP([3, 32, 1], bn=False, instance_norm=instance_norm))
            self.ad_feats_mlps.append(pt_utils.SharedMLP([neigh_ch, 1], bn=False, instance_norm=instance_norm))
            self.xyz_mlps.append(pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=instance_norm))
            
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None, query_features: torch.Tensor=None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features, idn = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            # idn(B, npoints, nsamples)
            # (B, npoints)
            
            idn_sum = torch.sum(idn, dim=2)
            
            idn_mask = (idn_sum > 0).unsqueeze(-1).float()
                        
            query_new_features = query_features.unsqueeze(-1)
            
            query_new_features = self.query_feats_mlps[i](query_new_features)
            
            idn_index = torch.nonzero(idn_mask)  
            
            _idn = torch.cuda.IntTensor(idn.shape[0], idn.shape[1], idn.shape[2]).zero_() + 1
            
            _idn[idn_index[:, 0], idn_index[:, 1], :] = idn[idn_index[:, 0], idn_index[:, 1], :]
            
            _idn = _idn.unsqueeze(1).float()
            
            new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
            
            new_features = new_features[:, 3:, :, :]
            
            feats_dist = query_new_features - new_features

            feats_dist = self.ad_feats_mlps[i](feats_dist)
            
            feats_dist_weights = F.softmax(feats_dist, dim=-1)

            space_dist = self.ad_xyz_mlps[i](new_features_xyz)

            space_dist_weights = F.softmax(feats_dist, dim=-1)            

            bi_weights = feats_dist_weights * space_dist_weights

            bi_edge = feats_dist * space_dist  # (B, ch, npoints, nsample)

            dist = torch.norm(new_features_xyz, dim=1, keepdim=True)        # (B, 1, npoints, nsample)
            
            dist_recip = 1.0 / (dist + 1e-8)                                # (B, 1, npoints, nsample)
                        
            dist_recip /= _idn
            
            # print(dist_recip.squeeze(1)[0, 0], _idn.squeeze(1)[0, 0])
            
            norm = torch.sum(dist_recip, dim=3, keepdim=True)               # (B, 1, npoints, 1)
            
            weights = dist_recip / norm                                     # (B, 1, npoints, nsamples)

            weights = weights * idn_mask.unsqueeze(1)
            
            # new_features[:, 3:, :, :] = weight * new_features[:, 3:, :, :]  # (B, 1, npoints, 1)
                       
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            new_features = weights * new_features 
            
            new_features = new_features * bi_edge
            
            new_features_xyz = self.xyz_mlps[i](new_features_xyz)
            
            new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
            
            new_features = new_features.sum(dim=3, keepdim=True)
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            
            new_features_xyz = new_features_xyz.squeeze(-1)
            
            new_features  = torch.cat([new_features_xyz, new_features], dim=1)
            
            new_features_list.append(new_features)
            
        return new_xyz, torch.cat(new_features_list, dim=1)


class Grouper7(nn.Module):
    def __init__(self, radius, nsamples, mlps, query_ch, neigh_ch, use_xyz=True, bn=False, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.xyz_mlps = nn.ModuleList()
        self.query_feats_mlps = nn.ModuleList()
        self.ad_xyz_mlps = nn.ModuleList()
        self.ad_feats_mlps = nn.ModuleList()
        self.neighbor_feats_mlps = nn.ModuleList()
        xyz_mlps = [[3, 32]]
        self.pool_methods = pool_methods
        self.radius = radius[0]

        for i in range(radius.__len__()):
            self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius[i], 
                                                nsamples[i], 
                                                use_xyz=use_xyz))
            
            mlp_spec = mlps[i]
            
            xyz_mlp_spec = xyz_mlps[i]
            
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=False, instance_norm=False))
            
            # make query features  channel is consistent wit neightbouring' channel
            self.query_feats_mlps.append(pt_utils.SharedMLP([query_ch, 32], bn=False, instance_norm=False))
            
            self.neighbor_feats_mlps.append(pt_utils.SharedMLP([neigh_ch, 32], bn=False, instance_norm=False))

            # for softmax in space dist
            self.ad_xyz_mlps.append(pt_utils.SharedMLP([3, 32, 1], bn=False, instance_norm=True, activation=None))
            
            # for softmax in feature dist
            self.ad_feats_mlps.append(pt_utils.SharedMLP([32, 1], bn=False, instance_norm=True, activation=None))
            
            self.xyz_mlps.append(pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=False))
            
        # print(self.ad_xyz_mlps[0], self.ad_feats_mlps[0])
    
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None, query_features: torch.Tensor=None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features, idn = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            # idn(B, npoints, nsamples)
            # (B, npoints)
            # print(xyz[])
            idn_sum = torch.sum(idn, dim=2)
            
            idn_mask = (idn_sum > 0).unsqueeze(-1).float()
            
            query_new_features = query_features.unsqueeze(-1)
            
            query_new_features = self.query_feats_mlps[i](query_new_features)
            
            idn_index = torch.nonzero(idn_mask)  
            
            _idn = torch.cuda.IntTensor(idn.shape[0], idn.shape[1], idn.shape[2]).zero_() + 1
            
            _idn[idn_index[:, 0], idn_index[:, 1], :] = idn[idn_index[:, 0], idn_index[:, 1], :]
            
            _idn = _idn.unsqueeze(1).float()
            
            new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
            
            # print(new_features_xyz[0, :, 0, :].permute(1, 0), idn[0, 0, :])
            
            new_features = new_features[:, 3:, :, :]
            
            new_features = self.neighbor_feats_mlps[i](new_features)

            feats_dist = query_new_features - new_features

            feats_dist = self.ad_feats_mlps[i](feats_dist)                  # (B, 1, npoints, nsample)
            
            feats_dist_weights = self.adaptive_softmax(feats_dist, _idn)    # (B, 1, npoints, nsample)

            space_dist = self.ad_xyz_mlps[i](new_features_xyz)
            
            space_dist_weights = self.adaptive_softmax(space_dist, _idn)            

            bi_weights = feats_dist_weights * space_dist_weights
            
            dist = torch.norm(new_features_xyz, dim=1, keepdim=True)        # (B, 1, npoints, nsample)
            
            dist_recip = 1.0 / (dist + 1e-8)                                # (B, 1, npoints, nsample)
            
            dist_recip /= (_idn + 1e-8)
            
            norm = torch.sum(dist_recip, dim=3, keepdim=True)               # (B, 1, npoints, 1)
            
            weights = dist_recip / (norm + 1e-8)                            # (B, 1, npoints, nsamples)

            bi_weights = weights * bi_weights  # (B, 1, npoints, nsamples)
            
            normed_biweights = bi_weights / (torch.sum(bi_weights, dim=-1, keepdim=True) + 1e-8) 
            
            normed_biweights = normed_biweights * idn_mask.unsqueeze(1)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            new_features = normed_biweights * new_features 
            
            new_features_xyz = self.xyz_mlps[i](new_features_xyz)
            
            new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
            
            new_features = new_features.sum(dim=3, keepdim=True)
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            
            new_features_xyz = new_features_xyz.squeeze(-1)
            
            new_features = torch.cat([new_features_xyz, new_features], dim=1)
            
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)
    
    def adaptive_softmax(self, vals, idn):
        # print(vals.shape, idn.shape)
        exp_vals = torch.exp(vals)/(idn + 1e-8)
        softmax_denominator = torch.sum(exp_vals, dim=-1, keepdim=True) + 1e-8
        return exp_vals/softmax_denominator


class Grouper8(nn.Module):
    def __init__(self, radius, nsamples, mlps, query_ch, neigh_ch, use_xyz=True, bn=False, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.xyz_mlps = nn.ModuleList()
        self.query_feats_mlps = nn.ModuleList()
        self.ad_xyz_mlps = nn.ModuleList()
        self.ad_feats_mlps = nn.ModuleList()
        self.neighbor_feats_mlps = nn.ModuleList()
        xyz_mlps = [[3, 32]]
        self.pool_methods = pool_methods
        self.radius = radius[0]

        for i in range(radius.__len__()):
            self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius[i], 
                                                nsamples[i], 
                                                use_xyz=use_xyz))
            
            mlp_spec = mlps[i]
            
            xyz_mlp_spec = xyz_mlps[i]
            
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=False, instance_norm=True))
            
            # make query features  channel is consistent wit neightbouring' channel
            self.query_feats_mlps.append(pt_utils.SharedMLP([query_ch, 32], bn=False, instance_norm=True))
            
            self.neighbor_feats_mlps.append(pt_utils.SharedMLP([neigh_ch, 32], bn=False, instance_norm=True))
            
            # for softmax in space dist
            self.ad_xyz_mlps.append(pt_utils.SharedMLP([3, 32, 1], bn=False, instance_norm=True, activation=None))
            
            # for softmax in feature dist
            self.ad_feats_mlps.append(pt_utils.SharedMLP([32, 1], bn=False, instance_norm=True, activation=None))
            
            self.xyz_mlps.append(pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=True))
            
        # print(self.ad_xyz_mlps[0], self.ad_feats_mlps[0])
    
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None, query_features: torch.Tensor=None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features, idn = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            # idn(B, npoints, nsamples)
            # (B, npoints)
            # print(xyz[])
            idn_sum = torch.sum(idn, dim=2)
            
            idn_mask = (idn_sum > 0).unsqueeze(-1).float()
            
            query_new_features = query_features.unsqueeze(-1)
            
            query_new_features = self.query_feats_mlps[i](query_new_features)
            
            idn_index = torch.nonzero(idn_mask)  
            
            _idn = torch.cuda.IntTensor(idn.shape[0], idn.shape[1], idn.shape[2]).zero_() + 1
            
            _idn[idn_index[:, 0], idn_index[:, 1], :] = idn[idn_index[:, 0], idn_index[:, 1], :]
            
            _idn = _idn.unsqueeze(1).float()
            
            new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
            
            # print(new_features_xyz[0, :, 0, :].permute(1, 0), idn[0, 0, :])
            
            new_features = new_features[:, 3:, :, :]
            
            new_features = self.neighbor_feats_mlps[i](new_features)

            feats_dist = query_new_features - new_features

            feats_dist = self.ad_feats_mlps[i](feats_dist)                  # (B, 1, npoints, nsample)
            
            feats_dist_weights = self.adaptive_softmax(feats_dist, _idn)    # (B, 1, npoints, nsample)

            space_dist = self.ad_xyz_mlps[i](new_features_xyz)
            
            space_dist_weights = self.adaptive_softmax(space_dist, _idn)            

            bi_weights = feats_dist_weights * space_dist_weights
            
            dist = torch.norm(new_features_xyz, dim=1, keepdim=True)        # (B, 1, npoints, nsample)
            
            dist_recip = 1.0 / (dist + 1e-8)                                # (B, 1, npoints, nsample)
            
            dist_recip /= (_idn + 1e-8)
            
            norm = torch.sum(dist_recip, dim=3, keepdim=True)               # (B, 1, npoints, 1)
            
            weights = dist_recip / (norm + 1e-8)                            # (B, 1, npoints, nsamples)

            bi_weights = weights * bi_weights  # (B, 1, npoints, nsamples)
            
            normed_biweights = bi_weights / (torch.sum(bi_weights, dim=-1, keepdim=True) + 1e-8) 
            
            normed_biweights = normed_biweights * idn_mask.unsqueeze(1)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            new_features = normed_biweights * new_features 
            
            new_features_xyz = self.xyz_mlps[i](new_features_xyz)
            
            new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
            
            new_features = new_features.sum(dim=3, keepdim=True)
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            
            new_features_xyz = new_features_xyz.squeeze(-1)
            
            new_features  = torch.cat([new_features_xyz, new_features], dim=1)
            
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)
    
    def adaptive_softmax(self, vals, idn):
        # print(vals.shape, idn.shape)
        exp_vals = torch.exp(vals)/(idn + 1e-8)
        softmax_denominator = torch.sum(exp_vals, dim=-1, keepdim=True) + 1e-8
        return exp_vals/softmax_denominator


class Grouper9(nn.Module):
    def __init__(self, radius, nsamples, mlps, query_ch, neigh_ch, use_xyz=True, bn=False, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.xyz_mlps = nn.ModuleList()
        self.query_feats_mlps = nn.ModuleList()
        self.ad_xyz_mlps = nn.ModuleList()
        self.ad_feats_mlps = nn.ModuleList()
        self.neighbor_feats_mlps = nn.ModuleList()
        xyz_mlps = [[3, 32]]
        self.pool_methods = pool_methods
        self.radius = radius[0]

        for i in range(radius.__len__()):
            self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius[i], 
                                                nsamples[i], 
                                                use_xyz=use_xyz))
            
            mlp_spec = mlps[i]
            
            xyz_mlp_spec = xyz_mlps[i]
            
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=False, instance_norm=True))
            
            # make query features  channel is consistent wit neightbouring' channel
            self.query_feats_mlps.append(pt_utils.SharedMLP([query_ch, 32], bn=False, instance_norm=True))
            
            self.neighbor_feats_mlps.append(pt_utils.SharedMLP([neigh_ch, 32], bn=False, instance_norm=True))
            
            # for softmax in space dist
            self.ad_xyz_mlps.append(pt_utils.SharedMLP([3, 32, 1], bn=False, instance_norm=True, activation=None))
            
            # for softmax in feature dist
            self.ad_feats_mlps.append(pt_utils.SharedMLP([32, 1], bn=False, instance_norm=True, activation=None))
            
            self.xyz_mlps.append(pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=False))
            
        # print(self.ad_xyz_mlps[0], self.ad_feats_mlps[0])

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None, query_features: torch.Tensor=None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features, idn = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            # idn(B, npoints, nsamples)
            # (B, npoints)
            # print(xyz[])
            idn_sum = torch.sum(idn, dim=2)
            
            idn_mask = (idn_sum > 0).unsqueeze(-1).float()
            
            query_new_features = query_features.unsqueeze(-1)
            
            query_new_features = self.query_feats_mlps[i](query_new_features)
            
            idn_index = torch.nonzero(idn_mask)  
            
            _idn = torch.cuda.IntTensor(idn.shape[0], idn.shape[1], idn.shape[2]).zero_() + 1
            
            _idn[idn_index[:, 0], idn_index[:, 1], :] = idn[idn_index[:, 0], idn_index[:, 1], :]
            
            _idn = _idn.unsqueeze(1).float()
            
            new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
            
            # print(new_features_xyz[0, :, 0, :].permute(1, 0), idn[0, 0, :])
            
            new_features = new_features[:, 3:, :, :]
            
            new_features = self.neighbor_feats_mlps[i](new_features)

            feats_dist = query_new_features - new_features

            feats_dist = self.ad_feats_mlps[i](feats_dist)                  # (B, 1, npoints, nsample)
            
            feats_dist_weights = self.adaptive_softmax(feats_dist, _idn)    # (B, 1, npoints, nsample)

            space_dist = self.ad_xyz_mlps[i](new_features_xyz)
            
            space_dist_weights = self.adaptive_softmax(space_dist, _idn)            

            bi_weights = feats_dist_weights * space_dist_weights
            
            dist = torch.norm(new_features_xyz, dim=1, keepdim=True)        # (B, 1, npoints, nsample)
            
            dist_recip = 1.0 / (dist + 1e-8)                                # (B, 1, npoints, nsample)
            
            dist_recip /= (_idn + 1e-8)
            
            norm = torch.sum(dist_recip, dim=3, keepdim=True)               # (B, 1, npoints, 1)
            
            weights = dist_recip / (norm + 1e-8)                            # (B, 1, npoints, nsamples)

            bi_weights = weights * bi_weights  # (B, 1, npoints, nsamples)
            
            normed_biweights = bi_weights / (torch.sum(bi_weights, dim=-1, keepdim=True) + 1e-8) 
            
            normed_biweights = normed_biweights * idn_mask.unsqueeze(1)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            new_features = normed_biweights * new_features
            
            new_features_xyz = self.xyz_mlps[i](new_features_xyz)
            
            new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
            
            new_features = new_features.sum(dim=3, keepdim=True)
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            
            new_features_xyz = new_features_xyz.squeeze(-1)
            
            new_features = torch.cat([new_features_xyz, new_features], dim=1)
            
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)
        
    def adaptive_softmax(self, vals, idn):
        # print(vals.shape, idn.shape)
        exp_vals = torch.exp(vals)/(idn + 1e-8)
        softmax_denominator = torch.sum(exp_vals, dim=-1, keepdim=True) + 1e-8
        return exp_vals/softmax_denominator
    
class GrouperForGrids(nn.Module):
    def __init__(self, radius, nsamples, mlps, use_xyz=True, bn=False, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.xyz_mlps = nn.ModuleList()
        xyz_mlps = [[3, 32, 32]]
        self.pool_methods = pool_methods
        self.radius = radius[0]
        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.GridQueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            mlp_spec = mlps[i]
            xyz_mlp_spec = xyz_mlps[i]
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=False, instance_norm=instance_norm))
            self.xyz_mlps.append(pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=instance_norm))
    
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        '''
            xyz: N, 3
            new_xyz: B, M, 3
            return: new_features: B, C, N
                    new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            # print('xyz', xyz.shape)
            
            # print('new_xyz', new_xyz.shape)

            new_features, idn = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            
            # (B, npoints)
            idn_sum = torch.sum(idn, dim=2)
            
            idn_mask = (idn_sum > 0).unsqueeze(-1).float()
                        
            idn_index = torch.nonzero(idn_mask)
            
            _idn = torch.cuda.IntTensor(idn.shape[0], idn.shape[1], idn.shape[2]).zero_() + 1
            
            _idn[idn_index[:, 0], idn_index[:, 1], :] = idn[idn_index[:, 0], idn_index[:, 1], :]
            
            _idn = _idn.unsqueeze(1).float()
            
            new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
            
            new_features = new_features[:, 3:, :, :]
            
            dist = torch.norm(new_features_xyz, dim=1, keepdim=True)        # (B, 1, npoints, nsample)
            
            dist_recip = 1.0 / (dist + 1e-8)                                # (B, 1, npoints, nsample)
            
            # dist_recip = 1.0 / (dist + 1e-8)                              # (B, 1, npoints, nsample)
            
            dist_recip /= _idn
            
            # print(dist_recip.squeeze(1)[0, 0], _idn.squeeze(1)[0, 0])
            
            norm = torch.sum(dist_recip, dim=3, keepdim=True)               # (B, 1, npoints, 1)
            
            weights = dist_recip / norm                                     # (B, 1, npoints, nsamples)

            weights = weights * idn_mask.unsqueeze(1)
            
            # new_features[:, 3:, :, :] = weight * new_features[:, 3:, :, :]  # (B, 1, npoints, 1)
            
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            
            new_features = weights * new_features

            new_features_xyz = self.xyz_mlps[i](new_features_xyz)
            
            new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
            
            new_features = new_features.sum(dim=3, keepdim=True)
            
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            
            new_features_xyz = new_features_xyz.squeeze(-1) * idn_mask.unsqueeze(1).squeeze(-1)
            
            # print('new_features_xyz', new_features_xyz.shape, 'idn_mask', idn_mask.shape, 'new_features', new_features.shape)
            
            new_features = torch.cat([new_features_xyz, new_features], dim=1)
            
            new_features_list.append(new_features)
            
        return new_xyz, torch.cat(new_features_list, dim=1)

class GrouperxyzForGrids(nn.Module):
    def __init__(self, radius, nsamples, use_xyz=True, bn=False, instance_norm=False, pool_methods=['max_pool']):
        super().__init__()
        self.groupers = nn.ModuleList()
        self.xyz_mlps = nn.ModuleList()
        xyz_mlps = [[3, 32, 32]]
        self.pool_methods = pool_methods
        self.radius = radius[0]
        for i in range(radius.__len__()):
            self.groupers.append(
                pointnet2_utils.GridQueryAndGroup(radius[i], nsamples[i], use_xyz=use_xyz)            
            )
            xyz_mlp_spec = xyz_mlps[i]
            self.xyz_mlps.append(pt_utils.SharedMLP(xyz_mlp_spec, bn=False, instance_norm=instance_norm))
            
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feature: torch.Tensor=None):
        '''
            xyz: B, N, 3
            new_features: B, C, N
            new_xyz: B, M, C
        '''
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features, idn = self.groupers[i](xyz, new_xyz)  # (B, C, npoint, nsample)
            # idn(B, npoints, nsamples)
            # (B, npoints)
            idn_sum = torch.sum(idn, dim=2)
            
            idn_mask = (idn_sum > 0).float()
                        
            new_features_xyz = new_features[:, :3, :, :]                    # (B, 3, npoints, nsample)
                        
            weights = idn_mask.unsqueeze(1)                                   # B, 1, npoints
            
            new_features_xyz = self.xyz_mlps[i](new_features_xyz)
            
            new_features_xyz = F.max_pool2d(new_features_xyz, kernel_size=[1, new_features_xyz.size(3)])
            
            new_features_xyz = new_features_xyz.squeeze(-1)  # B, 3, npoints
            # print('new_features_xyz.shape', new_features_xyz.shape)
            new_features_xyz = new_features_xyz * weights
            
            new_features_list.append(new_features_xyz)
            
        return new_xyz, torch.cat(new_features_list, dim=1)
