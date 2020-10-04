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
from dets.ops.pointnet2.layers_utils import Grouper7, Grouper8, Grouper9

class BiGNN(nn.Module):
    def __init__(self, **kwargs):
        super(BiGNN, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.use_voxel_feature = use_voxel_feature
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        -
        self.grouper_conv14 = Grouper7(radius=[1.0],
                                    nsamples=[64],
                                    mlps=[[32, 32]],
                                    use_xyz=True,
                                    query_ch=32,
                                    neigh_ch=32,
                                    bn=False)

        # 64 + 32
        self.conv14_structured_forward = partial(structured_forward,  grouper=self.grouper_conv14, 
                                    lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.05, 0.05, 0.1], offset=(0., -40., -3.))
        

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.grouper_conv24 = Grouper7(radius=[1.0],
                                        nsamples=[16],
                                        mlps=[[32, 32]],
                                        query_ch=32,
                                        neigh_ch=32,
                                        use_xyz=True,
                                        bn=False)
        
        # 64 + 32
        self.conv24_structured_forward = partial(structured_forward,  
                                                grouper=self.grouper_conv24, 
                                                lr_voxel_size=[0.4, 0.4, 1.0], 
                                                hr_voxel_size=[0.1, 0.1, 0.2], 
                                                offset=(0., -40., -3.))
                        

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(32, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.grouper_conv34 = Grouper7(radius=[1.0],
                                        nsamples=[16],
                                        mlps=[[32, 32]],
                                        query_ch=32,
                                        neigh_ch=64,
                                        use_xyz=True,
                                        bn=False)
                                    # pool_methods=cfg.MODEL.RPN.BACKBONE.POOLS)
        
        # 64 + 32
        self.conv34_structured_forward = partial(structured_forward,  
                                                grouper=self.grouper_conv34, 
                                                lr_voxel_size=[0.4, 0.4, 1.0], 
                                                hr_voxel_size=[0.2, 0.2, 0.4], 
                                                offset=(0., -40., -3.))

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            
        )
        
        last_pad = (0, 0, 0)

        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(32 + 32*3 + 32 * 3, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )

class BiGNN_V1(nn.Module):
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
            block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        
        self.grouper_conv14 = Grouper7(radius=[1.0],
                                        nsamples=[64],
                                        mlps=[[32, 32]],
                                        use_xyz=True,
                                        query_ch=32,
                                        neigh_ch=32,
                                        bn=False)
        
        # 64 + 32
        self.conv14_structured_forward = partial(structured_forward,  grouper=self.grouper_conv14, 
                                    lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.05, 0.05, 0.1], offset=(0., -40., -3.))
        

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.grouper_conv24 = Grouper7(radius=[1.0],
                                        nsamples=[16],
                                        mlps=[[32, 32]],
                                        query_ch=32,
                                        neigh_ch=32,
                                        use_xyz=True,
                                        bn=False)
        
        # 64 + 32
        self.conv24_structured_forward = partial(structured_forward,  
                                                grouper=self.grouper_conv24, 
                                                lr_voxel_size=[0.4, 0.4, 1.0], 
                                                hr_voxel_size=[0.1, 0.1, 0.2], 
                                                offset=(0., -40., -3.))
                        

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(32, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.grouper_conv34 = Grouper7(radius=[1.0],
                                        nsamples=[16],
                                        mlps=[[32, 32]],
                                        query_ch=32,
                                        neigh_ch=64,
                                        use_xyz=True,
                                        bn=False)
                                    # pool_methods=cfg.MODEL.RPN.BACKBONE.POOLS)
        
        # 64 + 32
        self.conv34_structured_forward = partial(structured_forward,  
                                                grouper=self.grouper_conv34, 
                                                lr_voxel_size=[0.4, 0.4, 1.0], 
                                                hr_voxel_size=[0.2, 0.2, 0.4], 
                                                offset=(0., -40., -3.))

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            
        )
        
        last_pad = (0, 0, 0)

        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(32 + 32*3 + 32 * 3, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
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
        rx_list = []
        x = self.conv_input(x)

        x_conv1 = self.conv1(x)
        rx_list.append(x_conv1)
        
        x_conv2 = self.conv2(x_conv1)
        rx_list.append(x_conv2)
        
        if self.use_voxel_feature:
            out_dict = {'vf2': x_conv2}

        x_conv3 = self.conv3(x_conv2)
        rx_list.append(x_conv3)

        x_conv4 = self.conv4(x_conv3)
        rx_list.append(x_conv4)

        structured_feats14 = self.conv14_structured_forward(rx_list, batch_size=kwargs['batch_size'])

        structured_feats24 = self.conv24_structured_forward(lrx=x_conv4, hrx=x_conv2, batch_size=kwargs['batch_size'])

        structured_feats34 = self.conv34_structured_forward(lrx=x_conv4, hrx=x_conv3, batch_size=kwargs['batch_size'])

        structured_conv4 = self.cat_sparse_features([structured_feats14, structured_feats24, structured_feats34], x_conv4)
        # for detection head        
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv4_out(structured_conv4)
        if self.use_voxel_feature:
            out_dict['vf1'] = out
        if self.use_voxel_feature:
            return out,  (None), out_dict
        else:
            return out, (None)

class BiGNN_V2(nn.Module):
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
            block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        
        self.grouper_conv14 = Grouper8(radius=[1.0],
                                        nsamples=[64],
                                        mlps=[[32, 32]],
                                        use_xyz=True,
                                        query_ch=32,
                                        neigh_ch=32,
                                        bn=False)
        
        # 64 + 32
        self.conv14_structured_forward = partial(structured_forward,  grouper=self.grouper_conv14, 
                                    lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.05, 0.05, 0.1], offset=(0., -40., -3.))
        

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.grouper_conv24 = Grouper8(radius=[1.0],
                                        nsamples=[32],
                                        mlps=[[32, 32]],
                                        query_ch=32,
                                        neigh_ch=32,
                                        use_xyz=True,
                                        bn=False)
        
        # 64 + 32
        self.conv24_structured_forward = partial(structured_forward,  
                                                grouper=self.grouper_conv24, 
                                                lr_voxel_size=[0.4, 0.4, 1.0], 
                                                hr_voxel_size=[0.1, 0.1, 0.2], 
                                                offset=(0., -40., -3.))
                        

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(32, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.grouper_conv34 = Grouper8(radius=[1.0],
                                        nsamples=[16],
                                        mlps=[[32, 32]],
                                        query_ch=32,
                                        neigh_ch=64,
                                        use_xyz=True,
                                        bn=False)
                                    # pool_methods=cfg.MODEL.RPN.BACKBONE.POOLS)
        
        # 64 + 32
        self.conv34_structured_forward = partial(structured_forward,  
                                                grouper=self.grouper_conv34, 
                                                lr_voxel_size=[0.4, 0.4, 1.0], 
                                                hr_voxel_size=[0.2, 0.2, 0.4], 
                                                offset=(0., -40., -3.))

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            
        )
        
        last_pad = (0, 0, 0)

        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(32 + 32*3 + 32 * 3, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
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

        x_conv1 = self.conv1(x)

        x_conv2 = self.conv2(x_conv1)
        if self.use_voxel_feature:
            out_dict = {'vf2': x_conv2}


        x_conv3 = self.conv3(x_conv2)

        x_conv4 = self.conv4(x_conv3)


        structured_feats14 = self.conv14_structured_forward(lrx=x_conv4, hrx=x_conv1, batch_size=kwargs['batch_size'])

        structured_feats24 = self.conv24_structured_forward(lrx=x_conv4, hrx=x_conv2, batch_size=kwargs['batch_size'])

        structured_feats34 = self.conv34_structured_forward(lrx=x_conv4, hrx=x_conv3, batch_size=kwargs['batch_size'])

        structured_conv4 = self.cat_sparse_features([structured_feats14, structured_feats24, structured_feats34], x_conv4)
        # for detection head        
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv4_out(structured_conv4)
        if self.use_voxel_feature:
            out_dict['vf1'] = out
        if self.use_voxel_feature:
            return out,  (None), out_dict
        else:
            return out, (None)

class BiGNN_V3(nn.Module):
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
            block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        
        self.grouper_conv14 = Grouper9(radius=[1.0],
                                        nsamples=[64],
                                        mlps=[[32, 32]],
                                        use_xyz=True,
                                        query_ch=32,
                                        neigh_ch=32,
                                        bn=False)
        
        # 64 + 32
        self.conv14_structured_forward = partial(structured_forward,  grouper=self.grouper_conv14, 
                                    lr_voxel_size=[0.4, 0.4, 1.0], hr_voxel_size=[0.05, 0.05, 0.1], offset=(0., -40., -3.))
        

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.grouper_conv24 = Grouper9(radius=[1.0],
                                        nsamples=[32],
                                        mlps=[[32, 32]],
                                        query_ch=32,
                                        neigh_ch=32,
                                        use_xyz=True,
                                        bn=False)
        
        # 64 + 32
        self.conv24_structured_forward = partial(structured_forward,  
                                                grouper=self.grouper_conv24, 
                                                lr_voxel_size=[0.4, 0.4, 1.0], 
                                                hr_voxel_size=[0.1, 0.1, 0.2], 
                                                offset=(0., -40., -3.))
                        

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(32, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.grouper_conv34 = Grouper9(radius=[1.0],
                                        nsamples=[16],
                                        mlps=[[32, 32]],
                                        query_ch=32,
                                        neigh_ch=64,
                                        use_xyz=True,
                                        bn=False)
                                    # pool_methods=cfg.MODEL.RPN.BACKBONE.POOLS)
        
        # 64 + 32
        self.conv34_structured_forward = partial(structured_forward,  
                                                grouper=self.grouper_conv34, 
                                                lr_voxel_size=[0.4, 0.4, 1.0], 
                                                hr_voxel_size=[0.2, 0.2, 0.4], 
                                                offset=(0., -40., -3.))

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            
        )
        
        last_pad = (0, 0, 0)

        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(32 + 32*3 + 32 * 3, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
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

        x_conv1 = self.conv1(x)

        x_conv2 = self.conv2(x_conv1)
        if self.use_voxel_feature:
            out_dict = {'vf2': x_conv2}
        
        x_conv3 = self.conv3(x_conv2)

        x_conv4 = self.conv4(x_conv3)

        structured_feats14 = self.conv14_structured_forward(lrx=x_conv4, hrx=x_conv1, batch_size=kwargs['batch_size'])

        structured_feats24 = self.conv24_structured_forward(lrx=x_conv4, hrx=x_conv2, batch_size=kwargs['batch_size'])

        structured_feats34 = self.conv34_structured_forward(lrx=x_conv4, hrx=x_conv3, batch_size=kwargs['batch_size'])

        structured_conv4 = self.cat_sparse_features([structured_feats14, structured_feats24, structured_feats34], x_conv4)
        # for detection head        
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv4_out(structured_conv4)
        if self.use_voxel_feature:
            out_dict['vf1'] = out
        if self.use_voxel_feature:
            return out,  (None), out_dict
        else:
            return out, (None)

