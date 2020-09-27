import spconv
from torch import nn
from ..utils import change_default_args, Sequential
from mmdet.ops.pointnet2 import pointnet2_utils
import torch
from dets.tools.ops import pts_in_boxes3d
from mmdet.core.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss
from mmdet.core import tensor2points
import torch.nn.functional as F
from functools import partial
from dets.tools.ops.pointnet2.layers_utils import Grouper4, Grouper5, Grouper6
from .bignn import BiGNN_V1, BiGNN_V2, BiGNN_V3
def post_act_block(in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                    conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        m = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
        )
    elif conv_type == 'spconv':
        m = spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, 
                            kernel_size, stride=stride, padding=padding,
                            bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
        )
    elif conv_type == 'inverseconv':
        m = spconv.SparseSequential(
            spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                        indice_key=indice_key, bias=False),
            norm_fn(out_channels),
            nn.ReLU(),
        )
    else:
        raise NotImplementedError
    return m

def structured_forward(lrx, hrx, batch_size, grouper, lr_voxel_size, hr_voxel_size, offset, cat_original=True):
        lr_indices = lrx.indices.float()
        hr_indices = hrx.indices.float()
        lr_voxel_size = torch.Tensor(lr_voxel_size).to(lr_indices.device)
        hr_voxel_size = torch.Tensor(hr_voxel_size).to(hr_indices.device)
        offset = torch.Tensor(offset).to(hr_indices.device)

        lr_indices[:, 1:] = lr_indices[:, [3, 2, 1]] * lr_voxel_size + \
                        offset + .5 * lr_voxel_size

        hr_indices[:, 1:] = hr_indices[:, [3, 2, 1]] * hr_voxel_size + \
                        offset + .5 * hr_voxel_size
        
        hr_features = hrx.features
        lr_features = lrx.features
        new_lr_features = []

        for bidx in range(batch_size):
            lr_mask = lr_indices[:, 0] == bidx
            hr_mask = hr_indices[:, 0] == bidx
            
            cur_lr_indices = lr_indices[lr_mask]
            cur_hr_indices = hr_indices[hr_mask]
            # cur_lr_features = lr_features[lr_mask]
            cur_hr_features = hr_features[hr_mask].unsqueeze(0).transpose(1, 2)

            cur_lr_xyz = cur_lr_indices[:, 1:].unsqueeze(0)
            cur_hr_xyz = cur_hr_indices[:, 1:].unsqueeze(0)
            _, new_features = grouper(cur_hr_xyz.contiguous(), cur_lr_xyz.contiguous(), cur_hr_features.contiguous())
            new_lr_features.append(new_features.squeeze(0))

        new_lr_features = torch.cat(new_lr_features, dim=1)
        new_lr_features = new_lr_features.transpose(0, 1)
        
        # print(lr_features.mean(dim=0), new_lr_features.mean(dim=0))
        lrx.features = torch.cat([lr_features, new_lr_features], dim=-1)
        return lrx

def structured_forward2(lrx, hrx, batch_size, grouper, lr_voxel_size, hr_voxel_size, offset, cat_original=True):
        lr_indices = lrx.indices.float()
        hr_indices = hrx.indices.float()
        lr_voxel_size = torch.Tensor(lr_voxel_size).to(lr_indices.device)
        hr_voxel_size = torch.Tensor(hr_voxel_size).to(hr_indices.device)
        offset = torch.Tensor(offset).to(hr_indices.device)

        lr_indices[:, 1:] = lr_indices[:, [3, 2, 1]] * lr_voxel_size + \
                        offset + .5 * lr_voxel_size

        hr_indices[:, 1:] = hr_indices[:, [3, 2, 1]] * hr_voxel_size + \
                        offset + .5 * hr_voxel_size
        
        hr_features = hrx.features
        lr_features = lrx.features
        new_lr_features = []
        features = []
        for bidx in range(batch_size):
            lr_mask = lr_indices[:, 0] == bidx
            hr_mask = hr_indices[:, 0] == bidx
            
            cur_lr_indices = lr_indices[lr_mask]
            cur_hr_indices = hr_indices[hr_mask]
            cur_lr_features = lr_features[lr_mask].unsqueeze(0).transpose(1, 2)
            cur_hr_features = hr_features[hr_mask].unsqueeze(0).transpose(1, 2)

            cur_lr_xyz = cur_lr_indices[:, 1:].unsqueeze(0)
            cur_hr_xyz = cur_hr_indices[:, 1:].unsqueeze(0)
            _, new_features = grouper(cur_hr_xyz.contiguous(), cur_lr_xyz.contiguous(), \
                cur_hr_features.contiguous(), cur_lr_features.contiguous())
            
            new_lr_features.append(new_features.squeeze(0))

        new_lr_features = torch.cat(new_lr_features, dim=1)
        # print(new_lr_features.shape)
        new_lr_features = new_lr_features.transpose(0, 1)
        
        # print(lr_features.mean(dim=0), new_lr_features.mean(dim=0))
        features = torch.cat([lr_features, new_lr_features], dim=-1)
        return features

class PCDetNeck(nn.Module):
    
    def __init__(self, output_shape,                  
                num_input_features=4,
                num_hidden_features=128):
        super().__init__()
        self.sparse_shape = output_shape

        self.backbone = PCDet3DNet(num_input_features)

        self.fcn = PCDetBEVNet(in_features=num_hidden_features, num_filters=256)

    
    def forward(self, voxel_features, coors, batch_size, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """        
        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x, point_misc = self.backbone(x, points_mean, is_test)

        x = x.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)

        x = self.fcn(x)

        if is_test:
            return x
        return x, point_misc        

class PCDetNeckImp(nn.Module):
    
    def __init__(self, output_shape,                  
                num_input_features=4,
                num_hidden_features=128,
                add_attention=False,
                use_hier_warp=False,
                use_voxel_feature=False
                ):
        super().__init__()
        self.sparse_shape = output_shape
        self.use_voxel_feature = use_voxel_feature

        self.backbone = PCDet3DImpro(num_input_features)
        
        self.fcn = PCDetBEVNet2(in_features=num_hidden_features, num_filters=256)
        self.add_attention = add_attention
        self.use_hier_warp = use_hier_warp

    def forward(self, voxel_features, coors, batch_size, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """        
        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]
        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x, point_misc = self.backbone(x, points_mean, is_test)

        x = x.dense()
        
        N, C, D, H, W = x.shape
        
        if self.add_attention:
            x_w = torch.sum(x, dim=1) # N, D, H, W
            x_w = nn.functional.softmax(x_w, 1) # N, D, H, W
        
            # print('x_w', x_w.shape)
            x_w = x_w.unsqueeze(1)
            x = x * x_w
        

        if self.use_hier_warp:    
            out_3d = x.clone()

        x = x.view(N, C * D, H, W)

        x = self.fcn(x)


        # if 
        if self.use_hier_warp:    
            if is_test:
                return x, out_3d
            else:
                return x, out_3d, point_misc     
        else:
            if is_test:
                return x
            else:
                return x, point_misc     

class CBiGNN(nn.Module):
    def __init__(self, output_shape,                  
                num_input_features=4,
                num_hidden_features=128,
                add_attention=False,
                use_hier_warp=False,
                use_voxel_feature=False,
                backbone='BiGNN_V1'):
        
        super().__init__()
        backbone_dict = {'BiGNN_V1':BiGNN_V1,  'BiGNN_V2':BiGNN_V2, 'BiGNN_V3': BiGNN_V3}
        self.sparse_shape = output_shape
        self.add_attention = add_attention    
        self.use_hier_warp = use_hier_warp
        self.use_voxel_feature = use_voxel_feature
        self.backbone = backbone_dict[backbone](num_input_features, use_voxel_feature)
        self.fcn = PCDetBEVNet2(in_features=num_hidden_features, num_filters=256)

    def forward(self, voxel_features, coors, batch_size, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """        
        points_mean = torch.zeros_like(voxel_features)
        
        points_mean[:, 0] = coors[:, 0]
        
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # print(x.indices[0], x.features[0])
        
        if self.use_voxel_feature:
            spatial_dict = {
                    'pc': x
                    }
        
        if self.use_voxel_feature:
            x, point_misc, vf_dict = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)
            spatial_dict.update(vf_dict)
        else:
            x, point_misc = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)

        x = x.dense()
        
        N, C, D, H, W = x.shape
        
        # print(x.shape)

        if self.add_attention:
            x_w = torch.sum(x, dim=1) # N, D, H, W
            x_w = nn.functional.softmax(x_w, 1) # N, D, H, W
            x_w = x_w.unsqueeze(1)
            x = x * x_w

        if self.use_hier_warp:    
            out_3d = x.clone()
        
        x = x.view(N, C * D, H, W)
        x = self.fcn(x)


        # if 
        # print('self.use_voxel_feature', self.use_voxel_feature)
        if self.use_hier_warp:    
            if is_test:
                return x, out_3d
            else:
                return x, out_3d, point_misc     
        elif self.use_voxel_feature:
            if is_test:
                return x, spatial_dict
            else:
                # print('spatial_dict')
                return x, spatial_dict, point_misc     
        else:
            if is_test:
                return x
            else:
                return x, point_misc   


class VANNeck(nn.Module):
    def __init__(self, output_shape,                  
                num_input_features=4,
                num_hidden_features=128,
                add_attention=False,
                use_hier_warp=False,
                use_voxel_feature=False):
        super().__init__()
        self.sparse_shape = output_shape

        
        self.add_attention = add_attention
        
        self.use_hier_warp = use_hier_warp
        self.use_voxel_feature = use_voxel_feature
        self.backbone = StructuredBackBone8xMRS(num_input_features, use_voxel_feature)
        self.fcn = PCDetBEVNet2(in_features=num_hidden_features, num_filters=256)

    def forward(self, voxel_features, coors, batch_size, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """        
        points_mean = torch.zeros_like(voxel_features)
        
        points_mean[:, 0] = coors[:, 0]
        
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # print(x.indices[0], x.features[0])
        
        if self.use_voxel_feature:
            spatial_dict = {
                    'pc': x
                    }
        
        if self.use_voxel_feature:
            x, point_misc, vf_dict = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)
            spatial_dict.update(vf_dict)
        else:
            x, point_misc = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)

        x = x.dense()
        
        N, C, D, H, W = x.shape
        
        # print(x.shape)

        if self.add_attention:
            x_w = torch.sum(x, dim=1) # N, D, H, W
            x_w = nn.functional.softmax(x_w, 1) # N, D, H, W
            # print('x_w', x_w.shape)
            x_w = x_w.unsqueeze(1)
            x = x * x_w
        
        if self.use_hier_warp:    
            out_3d = x.clone()
        
        x = x.view(N, C * D, H, W)
        x = self.fcn(x)


        # if 
        # print('self.use_voxel_feature', self.use_voxel_feature)
        if self.use_hier_warp:    
            if is_test:
                return x, out_3d
            else:
                return x, out_3d, point_misc     
        elif self.use_voxel_feature:
            if is_test:
                return x, spatial_dict
            else:
                # print('spatial_dict')
                return x, spatial_dict, point_misc     
        else:
            if is_test:
                return x
            else:
                return x, point_misc     

class VANNeck3(nn.Module):
    def __init__(self, output_shape,                  
                num_input_features=4,
                num_hidden_features=128,
                add_attention=False,
                use_hier_warp=False,
                use_voxel_feature=False):
        super().__init__()
        self.sparse_shape = output_shape

        
        self.add_attention = add_attention
        
        self.use_hier_warp = use_hier_warp
        self.use_voxel_feature = use_voxel_feature
        self.backbone = StructuredBackBone8xMRS2(num_input_features, use_voxel_feature)
        self.fcn = PCDetBEVNet2(in_features=num_hidden_features, num_filters=256)

    def forward(self, voxel_features, coors, batch_size, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """        
        points_mean = torch.zeros_like(voxel_features)
        
        points_mean[:, 0] = coors[:, 0]
        
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # print(x.indices[0], x.features[0])
        
        if self.use_voxel_feature:
            spatial_dict = {
                    'pc': x
                    }
        
        if self.use_voxel_feature:
            x, point_misc, vf_dict = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)
            spatial_dict.update(vf_dict)
        else:
            x, point_misc = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)

        x = x.dense()
        
        N, C, D, H, W = x.shape
        
        # print(x.shape)

        if self.add_attention:
            x_w = torch.sum(x, dim=1) # N, D, H, W
            x_w = nn.functional.softmax(x_w, 1) # N, D, H, W
            # print('x_w', x_w.shape)
            x_w = x_w.unsqueeze(1)
            x = x * x_w
        
        if self.use_hier_warp:    
            out_3d = x.clone()
        
        x = x.view(N, C * D, H, W)
        x = self.fcn(x)


        # if 
        # print('self.use_voxel_feature', self.use_voxel_feature)
        if self.use_hier_warp:    
            if is_test:
                return x, out_3d
            else:
                return x, out_3d, point_misc     
        elif self.use_voxel_feature:
            if is_test:
                return x, spatial_dict
            else:
                # print('spatial_dict')
                return x, spatial_dict, point_misc     
        else:
            if is_test:
                return x
            else:
                return x, point_misc   

class VANNeck4(nn.Module):
    def __init__(self, output_shape,                  
                num_input_features=4,
                num_hidden_features=128,
                add_attention=False,
                use_hier_warp=False,
                use_voxel_feature=False):
        super().__init__()
        self.sparse_shape = output_shape

        
        self.add_attention = add_attention
        
        self.use_hier_warp = use_hier_warp
        self.use_voxel_feature = use_voxel_feature
        self.backbone = StructuredBackBone8xMRS3(num_input_features, use_voxel_feature)
        self.fcn = PCDetBEVNet2(in_features=num_hidden_features, num_filters=256)

    def forward(self, voxel_features, coors, batch_size, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """        
        points_mean = torch.zeros_like(voxel_features)
        
        points_mean[:, 0] = coors[:, 0]
        
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # print(x.indices[0], x.features[0])
        
        if self.use_voxel_feature:
            spatial_dict = {
                    'pc': x
                    }
        
        if self.use_voxel_feature:
            x, point_misc, vf_dict = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)
            spatial_dict.update(vf_dict)
        else:
            x, point_misc = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)

        x = x.dense()
        
        N, C, D, H, W = x.shape
        
        # print(x.shape)

        if self.add_attention:
            x_w = torch.sum(x, dim=1) # N, D, H, W
            x_w = nn.functional.softmax(x_w, 1) # N, D, H, W
            x_w = x_w.unsqueeze(1)
            x = x * x_w

        if self.use_hier_warp:    
            out_3d = x.clone()
        
        x = x.view(N, C * D, H, W)
        x = self.fcn(x)


        # if 
        # print('self.use_voxel_feature', self.use_voxel_feature)
        if self.use_hier_warp:    
            if is_test:
                return x, out_3d
            else:
                return x, out_3d, point_misc     
        elif self.use_voxel_feature:
            if is_test:
                return x, spatial_dict
            else:
                # print('spatial_dict')
                return x, spatial_dict, point_misc     
        else:
            if is_test:
                return x
            else:
                return x, point_misc   



class VANNeck2(nn.Module):
    def __init__(self, output_shape,                  
                num_input_features=4,
                num_hidden_features=128,
                add_attention=False,
                use_hier_warp=False,
                use_voxel_feature=False):
        super().__init__()
        self.sparse_shape = output_shape
        self.add_attention = add_attention
        self.use_hier_warp = use_hier_warp
        self.use_voxel_feature = use_voxel_feature
        self.backbone = StructuredBackBone8xMRS(num_input_features, use_voxel_feature)
        self.fcn = PCDetBEVNet3(in_features=num_hidden_features, num_filters=256)
        
    def forward(self, voxel_features, coors, batch_size, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """        
        points_mean = torch.zeros_like(voxel_features)
        
        points_mean[:, 0] = coors[:, 0]
        
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # print(x.indices[0], x.features[0])
        
        if self.use_voxel_feature:
            spatial_dict = {
                    'pc': x
                    }
        if self.use_voxel_feature:
            x, point_misc, vf_dict = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)
            spatial_dict.update(vf_dict)
        else:
            x, point_misc = self.backbone(x, points_mean, is_test=is_test, batch_size=batch_size)

        x = x.dense()
        
        N, C, D, H, W = x.shape
        
        # print(x.shape)

        if self.add_attention:
            x_w = torch.sum(x, dim=1) # N, D, H, W
            x_w = nn.functional.softmax(x_w, 1) # N, D, H, W
            # print('x_w', x_w.shape)
            x_w = x_w.unsqueeze(1)
            x = x * x_w
        
        if self.use_hier_warp:    
            out_3d = x.clone()
        
        x = x.view(N, C * D, H, W)
        x = self.fcn(x)


        # if 
        # print('self.use_voxel_feature', self.use_voxel_feature)
        if self.use_hier_warp:    
            if is_test:
                return x, out_3d
            else:
                return x, out_3d, point_misc     
        elif self.use_voxel_feature:
            if is_test:
                return x, spatial_dict
            else:
                # print('spatial_dict')
                return x, spatial_dict, point_misc     
        else:
            if is_test:
                return x
            else:
                return x, point_misc     



class SpMiddleFHD(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=4,
                 num_hidden_features=128,
                 ):

        super(SpMiddleFHD, self).__init__()

        # print(output_shape)
        self.sparse_shape = output_shape
        self.backbone = VxNet(num_input_features)
        self.fcn = BEVNet(in_features=num_hidden_features, num_filters=256)


    def _make_layer(self, conv2d, bachnorm2d, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            nn.ZeroPad2d(1),
            conv2d(inplanes, planes, 3, stride=stride),
            bachnorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(conv2d(planes, planes, 3, padding=1))
            block.add(bachnorm2d(planes))
            block.add(nn.ReLU())
        return block, planes

    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        center_offsets = list()
        pts_labels = list()
        for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i].cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz, boxes3d)
            pts_label = pts_in_flag.max(0)[0].byte()

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):

        N = len(gt_bboxes)

        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return dict(
            aux_loss_cls = aux_loss_cls * 0.9,
            aux_loss_reg = aux_loss_reg * 2,
        )

    def forward(self, voxel_features, coors, batch_size, is_test=False):

        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x, point_misc = self.backbone(x, points_mean, is_test)

        x = x.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)

        x = self.fcn(x)

        if is_test:
            return x

        return x, point_misc

class SpMiddleFHDNA(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=4,
                 num_hidden_features=128,
                 ):

        super(SpMiddleFHDNA, self).__init__()

        # print(output_shape)
        self.sparse_shape = output_shape
        self.backbone = NVxNet(num_input_features)
        self.fcn = BEVNet(in_features=num_hidden_features, num_filters=256)
    
    def _make_layer(self, conv2d, bachnorm2d, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            nn.ZeroPad2d(1),
            conv2d(inplanes, planes, 3, stride=stride),
            bachnorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(conv2d(planes, planes, 3, padding=1))
            block.add(bachnorm2d(planes))
            block.add(nn.ReLU())
        return block, planes

    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        center_offsets = list()
        pts_labels = list()
        for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i].cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz, boxes3d)
            pts_label = pts_in_flag.max(0)[0].byte()

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):

        N = len(gt_bboxes)

        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return dict(
            aux_loss_cls = aux_loss_cls * 0.9,
            aux_loss_reg = aux_loss_reg * 2,
        )

    def forward(self, voxel_features, coors, batch_size, is_test=False):

        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x, point_misc = self.backbone(x, points_mean, is_test)

        x = x.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)

        x = self.fcn(x)

        if is_test:
            return x

        return x, point_misc

def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, 3, 2, padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
    )

def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    dist, idx = pointnet2_utils.three_nn(unknown, known)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    
    interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

    return interpolated_feats

class StructuredBackBone8xMRS(nn.Module):
    def __init__(self, num_input_features, use_voxel_feature=False):
        super().__init__() 
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.use_voxel_feature = use_voxel_feature
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.grouper_conv14 = Grouper4(radius=[1.0],
                                nsamples=[128],
                                mlps=[[16, 32]],
                                use_xyz=True,
                                bn=False)
        
        # 64 + 32
        self.conv14_structured_forward = partial(structured_forward,  grouper=self.grouper_conv14, 
                                        lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.05, 0.05, 0.1], offset=(0., -40., -3.))
        
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.grouper_conv24 = Grouper4(radius=[1.0],
                                    nsamples=[32],
                                    mlps=[[32, 32]],
                                    use_xyz=True,
                                    bn=False)
        
        # 'voxel_sizes': [[0.05, 0.05, 0.1],
        #                 [0.1, 0.1, 0.2],
        #                 [0.2, 0.2, 0.4], 
        #                 [0.4, 0.4, 0.8]],
        # 64 + 32
        self.conv24_structured_forward = partial(structured_forward,  grouper=self.grouper_conv24, 
                                        lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.1, 0.1, 0.2], offset=(0., -40., -3.))
                        

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.grouper_conv34 = Grouper4(radius=[1.0],
                                    nsamples=[16],
                                    mlps=[[64, 32]],
                                    use_xyz=True,
                                    bn=False)
                                    # pool_methods=cfg.MODEL.RPN.BACKBONE.POOLS)
        
        # 64 + 32
        self.conv34_structured_forward = partial(structured_forward,  grouper=self.grouper_conv34, 
                                        lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.2, 0.2, 0.4], offset=(0., -40., -3.))

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        last_pad = (0, 0, 0)

        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(64 + 96 + 32 + 32 + 32, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
    
    def cat_sparse_features(self, sparse_feats, x):
        xf_dim = x.features.shape[1]
        x.features = torch.cat([x.features] + [sf.features[:, xf_dim:]  for sf in sparse_feats], dim=1)
        return x


    def forward(self, x, points_mean, **kwargs):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """
        # str_fn = lambda x, n: "%s in backbone8x is %s" % (n , str(x.spatial_shape))

        x = self.conv_input(x)
        # print(self.conv_input.weight)
        # print(str_fn(x, "conv_input"))
        x_conv1 = self.conv1(x)
        # print(str_fn(x_conv1, "conv1"))
        x_conv2 = self.conv2(x_conv1)
        if self.use_voxel_feature:
            out_dict = {'vf2': x_conv2}

        # print(str_fn(x_conv2, "conv2"))
        x_conv3 = self.conv3(x_conv2)
        # print(str_fn(x_conv3, "x_conv3"))
        x_conv4 = self.conv4(x_conv3)
        # print(str_fn(x_conv4, "x_conv4"))
        
        # out_dict['vf1'] = x_conv4
        # x_conv4.features (n,c)
        # x_conv4.indices (n, 4)

        structured_feats14 = self.conv14_structured_forward(lrx=x_conv4, hrx=x_conv1, batch_size=kwargs['batch_size'])
        # print(x_conv4.indices[0])
        structured_feats24 = self.conv24_structured_forward(lrx=x_conv4, hrx=x_conv2, batch_size=kwargs['batch_size'])
        # print(x_conv4.indices[0])
        structured_feats34 = self.conv34_structured_forward(lrx=x_conv4, hrx=x_conv3, batch_size=kwargs['batch_size'])
        # print(x_conv4.indices[0])
        
        structured_conv4 = self.cat_sparse_features([structured_feats14, structured_feats24, structured_feats34], x_conv4)

        # print(structured_conv4.features.shape)

        # structured_feats = self.
        # print(str_fn(x_conv4, "conv4"))
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv4_out(structured_conv4)
        if self.use_voxel_feature:
            out_dict['vf1'] = out
        if self.use_voxel_feature:
            return out,  (None), out_dict
        else:

            return out, (None)

class StructuredBackBone8xMRS2(nn.Module):
    def __init__(self, num_input_features, use_voxel_feature=False):
        super().__init__() 
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.use_voxel_feature = use_voxel_feature
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.grouper_conv14 = Grouper5(radius=[1.0],
                                nsamples=[128],
                                mlps=[[16]],
                                use_xyz=True,
                                query_ch=32,
                                neigh_ch=16,
                                bn=False)
        
        # 64 + 32
        self.conv14_structured_forward = partial(structured_forward2,  grouper=self.grouper_conv14, 
                                        lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.05, 0.05, 0.1], offset=(0., -40., -3.))
        
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 16, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.grouper_conv24 = Grouper5(radius=[1.0],
                                    nsamples=[32],
                                    mlps=[[16]],
                                    query_ch=32,
                                    neigh_ch=16,
                                    use_xyz=True,
                                    bn=False)
        
        # 'voxel_sizes': [[0.05, 0.05, 0.1],
        #                 [0.1, 0.1, 0.2],
        #                 [0.2, 0.2, 0.4], 
        #                 [0.4, 0.4, 0.8]],
        # 64 + 32
        self.conv24_structured_forward = partial(structured_forward2,  grouper=self.grouper_conv24, 
                                        lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.1, 0.1, 0.2], offset=(0., -40., -3.))
                        

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(32, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.grouper_conv34 = Grouper5(radius=[1.0],
                                    nsamples=[16],
                                    mlps=[[16]],
                                    query_ch=32,
                                    neigh_ch=16,
                                    use_xyz=True,
                                    bn=False)
                                    # pool_methods=cfg.MODEL.RPN.BACKBONE.POOLS)
        
        # 64 + 32
        self.conv34_structured_forward = partial(structured_forward2,  grouper=self.grouper_conv34, 

                                        lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.2, 0.2, 0.4], offset=(0., -40., -3.))

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            
        )
        
        last_pad = (0, 0, 0)

        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(32 + 48 + 16 + 16 + 16, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        
    def cat_sparse_features(self, sparse_feats, x):
        xf_dim = x.features.shape[1]
        x.features = torch.cat([x.features] + [sf[:, xf_dim:] for sf in sparse_feats], dim=1)
        return x


    def forward(self, x, points_mean, **kwargs):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """
        # str_fn = lambda x, n: "%s in backbone8x is %s" % (n , str(x.spatial_shape))

        x = self.conv_input(x)
        # print(self.conv_input.weight)
        # print(str_fn(x, "conv_input"))
        x_conv1 = self.conv1(x)
        # print(str_fn(x_conv1, "conv1"))
        x_conv2 = self.conv2(x_conv1)
        if self.use_voxel_feature:
            out_dict = {'vf2': x_conv2}

        # print(str_fn(x_conv2, "conv2"))
        x_conv3 = self.conv3(x_conv2)
        # print(str_fn(x_conv3, "x_conv3"))
        x_conv4 = self.conv4(x_conv3)
        # print(str_fn(x_conv4, "x_conv4"))
        
        # out_dict['vf1'] = x_conv4
        # x_conv4.features (n,c)
        # x_conv4.indices (n, 4)

        structured_feats14 = self.conv14_structured_forward(lrx=x_conv4, hrx=x_conv1, batch_size=kwargs['batch_size'])
        # print(structured_feats14.features.shape, x_conv4.features.shape)
        # print(x_conv4.indices[0])
        structured_feats24 = self.conv24_structured_forward(lrx=x_conv4, hrx=x_conv2, batch_size=kwargs['batch_size'])
        # print(x_conv4.indices[0])
        structured_feats34 = self.conv34_structured_forward(lrx=x_conv4, hrx=x_conv3, batch_size=kwargs['batch_size'])

        # print(x_conv4.indices[0])
        
        structured_conv4 = self.cat_sparse_features([structured_feats14, structured_feats24, structured_feats34], x_conv4)

        # print(structured_conv4.features.shape)

        # structured_feats = self.
        # print(str_fn(x_conv4, "conv4"))
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv4_out(structured_conv4)
        if self.use_voxel_feature:
            out_dict['vf1'] = out
        if self.use_voxel_feature:
            return out,  (None), out_dict
        else:

            return out, (None)

class StructuredBackBone8xMRS3(nn.Module):
    def __init__(self, num_input_features, use_voxel_feature=False):
        super().__init__() 
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.use_voxel_feature = use_voxel_feature
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        
        # self.grouper_conv14 = Grouper6(radius=[1.0],
        #                         nsamples=[128],
        #                         mlps=[[16]],
        #                         use_xyz=True,
        #                         query_ch=32,
        #                         neigh_ch=16,
        #                         bn=False)
        

        # 64 + 32
        # self.conv14_structured_forward = partial(structured_forward2,  grouper=self.grouper_conv14, 
                                        # lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.05, 0.05, 0.1], offset=(0., -40., -3.))
        
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.grouper_conv34 = Grouper6(radius=[1.0],
                                    nsamples=[16],
                                    mlps=[[64, 32]],
                                    query_ch=64,
                                    neigh_ch=64,
                                    use_xyz=True,
                                    bn=False)
                                    # pool_methods=cfg.MODEL.RPN.BACKBONE.POOLS)
        
        # 64 + 32
        self.conv34_structured_forward = partial(structured_forward2,  grouper=self.grouper_conv34, 
                                        lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.2, 0.2, 0.4], offset=(0., -40., -3.))

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        
        last_pad = (0, 0, 0)

        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(64 + 32 + 32, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        
    def cat_sparse_features(self, sparse_feats, x):
        xf_dim = x.features.shape[1]
        x.features = torch.cat([x.features] + [sf[:, xf_dim:] for sf in sparse_feats], dim=1)
        return x


    def forward(self, x, points_mean, **kwargs):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """
        # str_fn = lambda x, n: "%s in backbone8x is %s" % (n , str(x.spatial_shape))

        x = self.conv_input(x)
        # print(self.conv_input.weight)
        # print(str_fn(x, "conv_input"))
        x_conv1 = self.conv1(x)
        # print(str_fn(x_conv1, "conv1"))
        x_conv2 = self.conv2(x_conv1)
        if self.use_voxel_feature:
            out_dict = {'vf2': x_conv2}

        # print(str_fn(x_conv2, "conv2"))
        x_conv3 = self.conv3(x_conv2)
        # print(str_fn(x_conv3, "x_conv3"))
        x_conv4 = self.conv4(x_conv3)
        # print(str_fn(x_conv4, "x_conv4"))
        
        # out_dict['vf1'] = x_conv4
        # x_conv4.features (n,c)
        # x_conv4.indices (n, 4)

        # structured_feats14 = self.conv14_structured_forward(lrx=x_conv4, hrx=x_conv1, batch_size=kwargs['batch_size'])
        # print(structured_feats14.features.shape, x_conv4.features.shape)
        # print(x_conv4.indices[0])
        # structured_feats24 = self.conv24_structured_forward(lrx=x_conv4, hrx=x_conv2, batch_size=kwargs['batch_size'])
        # print(x_conv4.indices[0])
        structured_feats34 = self.conv34_structured_forward(lrx=x_conv4, hrx=x_conv3, batch_size=kwargs['batch_size'])

        # print(x_conv4.indices[0])
        
        structured_conv4 = self.cat_sparse_features([structured_feats34], x_conv4)

        # print(structured_conv4.features.shape)

        # structured_feats = self.
        # print(str_fn(x_conv4, "conv4"))
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv4_out(structured_conv4)
        if self.use_voxel_feature:
            out_dict['vf1'] = out
        if self.use_voxel_feature:
            return out,  (None), out_dict
        else:

            return out, (None)

class PCDet3DNet(nn.Module):
    def __init__(self, num_input_features):
        super(PCDet3DNet, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        last_pad = (1, 0, 0)
        
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
    
    def forward(self, x, points_mean, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """
        # str_fn = lambda x, n: "%s in backbone8x is %s" % (n , str(x.spatial_shape))
        x = self.conv_input(x)
        # print(str_fn(x, "conv_input"))
        x_conv1 = self.conv1(x)
        # print(str_fn(x_conv1, "conv1"))
        x_conv2 = self.conv2(x_conv1)
        # print(str_fn(x_conv2, "conv2"))
        x_conv3 = self.conv3(x_conv2)
        # print(str_fn(x_conv3, "conv3"))
        x_conv4 = self.conv4(x_conv3)
        # print(str_fn(x_conv4, "conv4"))
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        # print(str_fn(out,  "out"))
        # out = out.dense()

        # N, C, D, H, W = out.shape
        # out = out.view(N, C * D, H, W)
        return out, (None)

class PCDet3DImpro(nn.Module):
    def __init__(self, num_input_features):
        super(PCDet3DImpro, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        last_pad = (0, 0, 0)
        
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 5]
            spconv.SparseConv3d(64, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )

    def forward(self, x, points_mean, is_test=False):    
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size: 
        :return:
        """
        # str_fn = lambda x, n: "%s in backbone8x is %s" % (n , str(x.spatial_shape))
        x = self.conv_input(x)
        # print(str_fn(x, "conv_input"))
        x_conv1 = self.conv1(x)
        # print(str_fn(x_conv1, "conv1"))
        x_conv2 = self.conv2(x_conv1)
        # print(str_fn(x_conv2, "conv2"))
        x_conv3 = self.conv3(x_conv2)
        # print(str_fn(x_conv3, "conv3"))
        x_conv4 = self.conv4(x_conv3)
        # print(str_fn(x_conv4, "conv4"))
        # for detection head
        # [200, 176, 5] -> [200, 176, 5]
        out = self.conv_out(x_conv4)

        # N, C, D, H, W = out.shape
        # out = out.view(N, C * D, H, W)
        return out, (None)

class VxNet(nn.Module):
    def __init__(self, num_input_features):
        super(VxNet, self).__init__()

        self.conv0 = double_conv(num_input_features, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 64, 'down2')

        self.conv3 = triple_conv(64, 64, 'subm3')  # middle line

        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (1, 1, 1), (1, 1, 1), bias=False),  # shape no change
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.point_fc = nn.Linear(160, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)
    
    def forward(self, x, points_mean, is_test=False):
            
        x = self.conv0(x)
        x = self.down0(x)  # sp
        x = self.conv1(x)  # 2x sub
        
        if not is_test:
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.1, .1, .2))
            p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

        x = self.down1(x)
        x = self.conv2(x)

        if not is_test:
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.2, .2, .4))
            p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

        x = self.down2(x)
        x = self.conv3(x)

        if not is_test:
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.4, .4, .8))
            p3 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

        out = self.extra_conv(x)

        if is_test:
            return out, None
        
        pointwise = self.point_fc(torch.cat([p1, p2, p3], dim=-1))
        point_cls = self.point_cls(pointwise)
        point_reg = self.point_reg(pointwise)
        return out, (points_mean, point_cls, point_reg)



class NVxNet(nn.Module):

    def __init__(self, num_input_features):
        super(NVxNet, self).__init__()

        self.conv0 = double_conv(num_input_features, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 64, 'down2')

        self.conv3 = triple_conv(64, 64, 'subm3')  # middle line

        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (1, 1, 1), (1, 1, 1), bias=False),  # shape no change
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        # self.point_fc = nn.Linear(160, 64, bias=False)
        # self.point_cls = nn.Linear(64, 1, bias=False)
        # self.point_reg = nn.Linear(64, 3, bias=False)
    
    def forward(self, x, points_mean, is_test=False):

        x = self.conv0(x)
        x = self.down0(x)  # sp
        x = self.conv1(x)  # 2x sub
        
        # if not is_test:
            # vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.1, .1, .2))
            # p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

        x = self.down1(x)
        x = self.conv2(x)

        # if not is_test:
        #     vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.2, .2, .4))
        #     p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

        x = self.down2(x)
        x = self.conv3(x)

        # if not is_test:
            # vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.4, .4, .8))
            # p3 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

        out = self.extra_conv(x)

        # if is_test:
            # return out, None

        # pointwise = self.point_fc(torch.cat([p1, p2, p3], dim=-1))
        # point_cls = self.point_cls(pointwise)
        # point_reg = self.point_reg(pointwise)
        return out, None

class BEVNet(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super(BEVNet, self).__init__()
        BatchNorm2d = change_default_args(
            eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)

        self.conv0 = Conv2d(in_features, num_filters, 3, padding=1)
        self.bn0 = BatchNorm2d(num_filters)

        self.conv1 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = BatchNorm2d(num_filters)

        self.conv2 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = BatchNorm2d(num_filters)

        self.conv3 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = BatchNorm2d(num_filters)

        self.conv4 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = BatchNorm2d(num_filters)

        self.conv5 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn5 = BatchNorm2d(num_filters)

        self.conv6 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn6 = BatchNorm2d(num_filters)

        self.conv7 = Conv2d(num_filters, num_filters, 1)
        self.bn7 = BatchNorm2d(num_filters)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv3(x)
        x = F.relu(self.bn3(x), inplace=True)
        x = self.conv4(x)
        x = F.relu(self.bn4(x), inplace=True)
        x = self.conv5(x)
        x = F.relu(self.bn5(x), inplace=True)
        x = self.conv6(x)
        x = F.relu(self.bn6(x), inplace=True)
        conv6 = x.clone()
        x = self.conv7(x)
        x = F.relu(self.bn7(x), inplace=True)
        return x, conv6

class PCDetBEVNet(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super().__init__()
        args = {
            'concat_input': False, 
            'num_input_features': in_features,
            'layer_nums': [5, 5],
            'layer_strides': [1, 2],
            'num_filters': [128, 256],
            'upsample_strides': [1, 2],
            'num_upsample_filters': [256, 256],
        }
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        # if args['use_norm']:
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = partial(nn.Conv2d, bias=False)
        ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        
        self.conv_out = Sequential(
                # nn.ZeroPad2d(1),
                Conv2d(c_in, num_filters, 1),
                BatchNorm2d(num_filters),
                nn.ReLU(),
            )
        

    def forward(self, x_in):
        ups = []
        x = x_in
        # ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)        
        
        # conv_ps = ups[0].clone()
        conv_ps = x.clone()

        x = self.conv_out(x)
        return x, conv_ps

class PCDetBEVNet2(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super().__init__()
        args = {
            'concat_input': False, 
            'num_input_features': in_features,
            'layer_nums': [5, 5],
            'layer_strides': [1, 2],
            'num_filters': [128, 256],
            'upsample_strides': [1, 2],
            'num_upsample_filters': [256, 256],
        }
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        # if args['use_norm']:
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = partial(nn.Conv2d, bias=False)
        ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        
        self.conv_out = Sequential(
                # nn.ZeroPad2d(1),
                Conv2d(c_in, num_filters, 1),
                BatchNorm2d(num_filters),
                nn.ReLU(),
            )
        

    def forward(self, x_in):
        ups = []
        x = x_in
        # ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)        
        
        conv_ps = ups[0].clone()

        x = self.conv_out(x)
        return x, conv_ps

class PCDetBEVNet3(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super().__init__()
        args = {
            'concat_input': False, 
            'num_input_features': in_features,
            'layer_nums': [5, 5],
            'layer_strides': [1, 2],
            'num_filters': [128, 256],
            'upsample_strides': [1, 2],
            'num_upsample_filters': [256, 256],
        }
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        # if args['use_norm']:
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = partial(nn.Conv2d, bias=False)
        ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU()
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        
        self.conv_out = Sequential(
                # nn.ZeroPad2d(1),
                Conv2d(c_in, num_filters, 1),
                BatchNorm2d(num_filters),
                nn.ReLU(),
            )
        

    def forward(self, x_in):
        ups = []
        x = x_in
        # ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]

        
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)        
        
        conv_ps = ups[0].detach()
        
        x = self.conv_out(x)
        return x, conv_ps
