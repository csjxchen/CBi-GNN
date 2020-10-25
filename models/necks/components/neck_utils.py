import spconv
import torch
import torch.nn as nn
from dets.ops.pointnet2 import pointnet2_utils

def post_act_block(in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                    conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        m = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size, stride=stride, bias=False, indice_key=indice_key),
            # spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
            
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

def parse_spconv_cfg(downsample_layer_cfgs, norm_fn):
    layers = nn.ModuleList()

    for layer_dict in downsample_layer_cfgs: 
        # for l in layers:
        assert len(layer_dict['types']) == len(layer_dict['indice_keys']), f"{len(layer_dict['types'])} == {len(layer_dict['indice_keys'])}?"
        assert len(layer_dict['types']) == (len(layer_dict['filters'])-1), f"{len(layer_dict['types'])} == {len(layer_dict['filters'])-1}?"
        _sequentials = []
        
        for i in range(len(layer_dict['types'])):
            _sequentials.append(post_act_block(
                                    layer_dict['filters'][i], 
                                    layer_dict['filters'][i + 1], 
                                    3, 
                                    stride=layer_dict['strides'][i],
                                    norm_fn=norm_fn, 
                                    padding=layer_dict['paddings'][i] if len(layer_dict['paddings'][i]) > 1 else layer_dict['paddings'][i][0], 
                                    conv_type=layer_dict['types'][i],
                                    indice_key=layer_dict['indice_keys'][i])
                                )
        layers.append(spconv.SparseSequential(*_sequentials))
    return layers

def structured_forward_v1(feats, lr_index, hr_index, batch_size, grouper, lr_voxel_size, hr_voxel_size, offset, shuffle=False):
        lrx = feats[lr_index]
        hrx = feats[hr_index]
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
            cur_hr_features = hr_features[hr_mask]
            if shuffle:
                random_perm = torch.randperm(cur_hr_indices.shape[0])
                cur_hr_features = cur_hr_features[random_perm]
                cur_hr_indices  = cur_hr_indices[random_perm]
            cur_hr_features = cur_hr_features.unsqueeze(0).transpose(1, 2)
            cur_lr_xyz = cur_lr_indices[:, 1:].unsqueeze(0)
            cur_hr_xyz = cur_hr_indices[:, 1:].unsqueeze(0)
            _, new_features = grouper(cur_hr_xyz.contiguous(), cur_lr_xyz.contiguous(), cur_hr_features.contiguous())
            new_lr_features.append(new_features.squeeze(0))
        
        new_lr_features = torch.cat(new_lr_features, dim=1)
        new_lr_features = new_lr_features.transpose(0, 1)
        # new_lr_features = torch.cat([lr_features, new_lr_features], dim=-1)
        # lrx.features = new_lr_features
        return new_lr_features

def fps_voxel_downscale_operation(x, voxel_size, offset, batch_size, downscaling=2.0, raw_sizes=None):
    """
        x:
        npoints: target npoints
        voxel_size: (x, y, z)
        offset: (x, y, z)
        sparse_shape: [z, y, x] in lidar coordinate

    """
    sparse_shape = x.spatial_shape
    indices = x.indices
    locs = indices.float()

    voxel_size = torch.Tensor(voxel_size).to(locs.device)
    offset = torch.Tensor(offset).to(locs.device)
    locs[:, 1:] = locs[:, [3, 2, 1]] * voxel_size + \
                        offset + .5 * voxel_size
    locs = locs[:, 1:]
    features = x.features
    sparser_indices  = []
    sparser_features = []
    if raw_sizes is None:
        _raw_sizes = []
    else:
        _raw_sizes = raw_sizes
    for bidx in range(batch_size):
        cur_mask = indices[:, 0] == bidx
        cur_locs = locs[cur_mask]
        cur_indices = indices[cur_mask]
        cur_features = features[cur_mask]
        randperm = torch.randperm(cur_indices.shape[0])
        cur_locs = cur_locs[randperm]
        cur_indices = cur_indices[randperm]
        cur_features = cur_features[randperm]
        
        if raw_sizes is None:
            _raw_sizes.append(cur_indices.shape[0])
        
        # cur_size = cur_indices.shape[0]
        cur_npoints = int(_raw_sizes[bidx] / downscaling)
        idx = pointnet2_utils.furthest_point_sample(cur_locs.unsqueeze(0).contiguous(), cur_npoints)
        new_indices = pointnet2_utils.gather_operation(cur_indices.float().unsqueeze(0).transpose(1,2).contiguous(), idx.contiguous())
        new_features = pointnet2_utils.gather_operation(cur_features.unsqueeze(0).transpose(1, 2).contiguous(), idx.contiguous())
        sparser_indices.append(new_indices.squeeze(0).transpose(0, 1).contiguous().long())
        sparser_features.append(new_features.squeeze(0).transpose(0, 1).contiguous())
    


    sparser_features = torch.cat(sparser_features, dim=0)
    sparser_indices = torch.cat(sparser_indices, dim=0)
    x = spconv.SparseConvTensor(sparser_features, sparser_indices, sparse_shape, batch_size)
    if raw_sizes is None:
        return x, _raw_sizes
    else:
        return x

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

def threeNN_sparser_to_densor(denserx, sparserx, denserx_voxel_size, sparser_voxel_size, offset, batch_size):
    denserx_indices = denserx.indices
    sparserx_indices = sparserx.indices

    denser_locs = denserx_indices.float()
    sparser_locs = sparserx_indices.float()

    denserx_voxel_size = torch.Tensor(denserx_voxel_size).to(denserx_indices.device)
    sparser_voxel_size = torch.Tensor(sparser_voxel_size).to(sparserx_indices.device)

    offset = torch.Tensor(offset).to(denserx_indices.device)
    denser_locs[:, 1:] = denser_locs[:, [3, 2, 1]] * denserx_voxel_size + \
                        offset + .5 * denserx_voxel_size
    sparser_locs[:, 1:] = sparser_locs[:, [3, 2, 1]] * sparser_voxel_size + \
                            offset + .5 * sparser_voxel_size    
    sparser_features = sparserx.features
    # denser_locs  = denser_locs[:, 1:]
    # sparser_locs  = sparser_locs[:, 1:]
    dist, idx = pointnet2_utils.three_nn(denser_locs, sparser_locs)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(sparser_features, idx, weight)
    # for bidx in range(batch_size):
    #     denser_mask = denserx_indices[:, 0] == bidx
    #     sparser_mask = sparserx_indices[:, 0] == bidx
    #     cur_denser_locs = denser_locs[denser_mask] 
    #     cur_sparser_locs = sparser_locs[sparser_mask] 
    #     cur_denser_features = denser_features[denser_mask]
    #     cur_sparser_features = sparser_features[sparser_mask]
    #     dist, idx = pointnet2_utils.three_nn(cur_denser_locs.unsqueeze(0).contiguous(), cur_sparser_locs.unsqueeze(0).contiguous())
    #     dist_recip = 1.0 / (dist + 1e-8)
    #     norm = torch.sum(dist_recip, dim=2, keepdim=True)
    #     weight = dist_recip / norm
    #     cur_interpolated_feats = pointnet2_utils.three_interpolate(cur_sparser_features.unsqueeze(0).transpose(1,2).contiguous(), idx, weight)
    #     interpolated_feats.append(cur_interpolated_feats.squeeze(0).transpose(0, 1).contiguous())    
    # interpolated_feats = torch.cat(interpolated_feats, dim=0)
    return interpolated_feats


# def structured_forward_v2(feats, lr_index, hr_index, batch_size, grouper, lr_voxel_size, hr_voxel_size, offset):
#         lrx = feats[lr_index]
#         hrx = feats[hr_index]
#         lr_indices = lrx.indices.float()
#         hr_indices = hrx.indices.float()
#         lr_voxel_size = torch.Tensor(lr_voxel_size).to(lr_indices.device)
#         hr_voxel_size = torch.Tensor(hr_voxel_size).to(hr_indices.device)
#         offset = torch.Tensor(offset).to(hr_indices.device)
#         lr_indices[:, 1:] = lr_indices[:, [3, 2, 1]] * lr_voxel_size + \
#                         offset + .5 * lr_voxel_size
#         hr_indices[:, 1:] = hr_indices[:, [3, 2, 1]] * hr_voxel_size + \
#                         offset + .5 * hr_voxel_size
#         hr_features = hrx.features
#         lr_features = lrx.features
#         new_lr_features = []
#         for bidx in range(batch_size):
#             lr_mask = lr_indices[:, 0] == bidx
#             hr_mask = hr_indices[:, 0] == bidx
#             cur_lr_indices = lr_indices[lr_mask]
#             cur_hr_indices = hr_indices[hr_mask]
#             cur_hr_features = hr_features[hr_mask].unsqueeze(0).transpose(1, 2)
#             cur_lr_xyz = cur_lr_indices[:, 1:].unsqueeze(0)
#             cur_hr_xyz = cur_hr_indices[:, 1:].unsqueeze(0)
#             _, new_features = grouper(cur_hr_xyz.contiguous(), cur_lr_xyz.contiguous(), cur_hr_features.contiguous())
#             new_lr_features.append(new_features.squeeze(0))
#         new_lr_features = torch.cat(new_lr_features, dim=1)
#         new_lr_features = new_lr_features.transpose(0, 1)
#         return new_lr_features