import spconv
import torch
import torch.nn as nn

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
def structured_forward(feats, lr_index, hr_index, batch_size, grouper, lr_voxel_size, hr_voxel_size, offset):
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
            cur_hr_features = hr_features[hr_mask].unsqueeze(0).transpose(1, 2)
            cur_lr_xyz = cur_lr_indices[:, 1:].unsqueeze(0)
            cur_hr_xyz = cur_hr_indices[:, 1:].unsqueeze(0)
            _, new_features = grouper(cur_hr_xyz.contiguous(), cur_lr_xyz.contiguous(), cur_hr_features.contiguous())
            new_lr_features.append(new_features.squeeze(0))
        new_lr_features = torch.cat(new_lr_features, dim=1)
        new_lr_features = new_lr_features.transpose(0, 1)
        # new_lr_features = torch.cat([lr_features, new_lr_features], dim=-1)
        return new_lr_features

