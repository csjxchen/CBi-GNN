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
# from dets.ops.pointnet2.layers_utils import Grouper7, Grouper8, Grouper9
from dets.ops.pointnet2 import grouper_models 
from .neck_utils import *
structured_forward_fn = {'v1': structured_forward_v1}
class BiGNN(nn.Module):
    def __init__(self, model_cfg):
        super(BiGNN, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.use_voxel_feature = use_voxel_feature
        self.model_cfg = model_cfg
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.model_cfg.conv_inputs[0],  self.model_cfg.conv_inputs[1], 3, padding=1, bias=False, indice_key='subm0'),
            norm_fn(self.model_cfg.conv_inputs[1]),
            nn.ReLU(),
        )
        block = post_act_block
        self.downsample_layers = nn.ModuleList()
        for layer_dict in self.model_cfg.downsample_layers: 
            # for l in layers:
            assert len(layer_dict['types']) == len(layer_dict['indice_keys']), f"{len(layer_dict['types'])} == {len(layer_dict['indice_keys'])}?"
            assert len(layer_dict['types']) == (len(layer_dict['filters'])-1), f"{len(layer_dict['types'])} == {len(layer_dict['filters'])-1}?"
            _sequentials = []
            
            for i in range(len(layer_dict['types'])):
                _sequentials.append(block(
                                    layer_dict['filters'][i], 
                                    layer_dict['filters'][i + 1], 
                                    3, 
                                    stride=layer_dict['strides'][i],
                                    norm_fn=norm_fn, 
                                    padding=layer_dict['paddings'][i] if len(layer_dict['paddings'][i]) > 1 else layer_dict['paddings'][i][0],
                                    conv_type=layer_dict['types'][i],
                                    indice_key=layer_dict['indice_keys'][i]))
            
            self.downsample_layers.append(spconv.SparseSequential(*_sequentials))                

        last_pad = (0, 0, 0)
        
        out_channels = self.model_cfg.downsample_layers[-1]['filters'][-1] + 64 * len(self.model_cfg.groupers)
        
        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(out_channels, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )

        self.groupers = nn.ModuleList()
        self.grouper_forward_fns = []
        for i in range(len(self.model_cfg.groupers)):
            grouper_dict = self.model_cfg.groupers[i]
            # print(grouper_dict)
            self.groupers.append(grouper_models[grouper_dict['grouper_type']](**grouper_dict.args))
            self.grouper_forward_fns.append(partial(structured_forward_fn[grouper_dict['forward_type']], grouper=self.groupers[-1],
                                            **grouper_dict['maps']))

    def forward(self, x, **kwargs):
        rx_list = []
        x = self.conv_input(x)

        for i, layer in enumerate(self.downsample_layers):
            # print(f"layer {i}")
            x = layer(x)
            rx_list.append(x)
        
        lrx_list = []        
        
        for gf_fn in self.grouper_forward_fns:
            _lrx = gf_fn(rx_list, batch_size=kwargs['batch_size'])
            lrx_list.append(_lrx)
        
        lrx = torch.cat([rx_list[-1].features, *lrx_list], dim=-1)
        rx_list[-1].features = lrx
        # [200, 176, 5]
        out = self.conv4_out(rx_list[-1])
        
        return out


class BiGNN_Submanifold(nn.Module):
    def __init__(self, model_cfg):
        super(BiGNN_Submanifold, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.use_voxel_feature = use_voxel_feature
        self.model_cfg = model_cfg
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.model_cfg.conv_inputs[0],  self.model_cfg.conv_inputs[1], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(self.model_cfg.conv_inputs[1]),
            nn.ReLU(),
        )
        block = post_act_block
        # self.downsample_layers = parse_spconv_cfg(self.model_cfg.downsample_layers)
        self.subm_layers = parse_spconv_cfg(self.model_cfg.subm_layers, norm_fn=norm_fn)              
        
        last_pad = (0, 0, 0)
        self.subm_out = spconv.SparseSequential(
            spconv.SparseConv3d(32 + 32*3 + 32 * 3, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        
        self.groupers = nn.ModuleList()
        self.grouper_forward_fns = []
        for i in range(len(self.model_cfg.groupers)):
            grouper_dict = self.model_cfg.groupers[i]
            # print(grouper_dict)
            self.groupers.append(grouper_models[grouper_dict['grouper_type']](**grouper_dict.args))
            self.grouper_forward_fns.append(partial(structured_forward, 
                                                    grouper=self.groupers[-1],
                                                    **grouper_dict['maps']))

    def forward(self, x, **kwargs):
        rx_list = []
        x = self.conv_input(x)
        for i, layer in enumerate(self.subm_layers):
            # print(f"layer {i}")
            x = layer(x)
            rx_list.append(x)

        # for i, layer in enumerate(self.):
        #     x = layer(x)
        #     rx_list.append(x)
        
        lrx_list = []        
        
        for gf_fn in self.grouper_forward_fns:
            _lrx = gf_fn(rx_list, batch_size=kwargs['batch_size'])
            lrx_list.append(_lrx)
        
        lrx = torch.cat([rx_list[-1].features, *lrx_list], dim=-1)
        rx_list[-1].features = lrx
        # [200, 176, 5] -> [200, 176, 2]
        out = self.subm_out(rx_list[-1])
        return out

class BiGNN_reproduce(nn.Module):
    def __init__(self, model_cfg):
        super(BiGNN_reproduce, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.use_voxel_feature = use_voxel_feature

        self.model_cfg = model_cfg
        self.repeat_num = self.model_cfg.repeat_num 
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.model_cfg.conv_inputs[0],  self.model_cfg.conv_inputs[1], 3, padding=1, bias=False, indice_key='subm0'),
            norm_fn(self.model_cfg.conv_inputs[1]),
            nn.ReLU(),
        )
        block = post_act_block
        self.downsample_layers = nn.ModuleList()
        for layer_dict in self.model_cfg.downsample_layers: 
            # for l in layers:
            assert len(layer_dict['types']) == len(layer_dict['indice_keys']), f"{len(layer_dict['types'])} == {len(layer_dict['indice_keys'])}?"
            assert len(layer_dict['types']) == (len(layer_dict['filters'])-1), f"{len(layer_dict['types'])} == {len(layer_dict['filters'])-1}?"
            _sequentials = []
            
            for i in range(len(layer_dict['types'])):
                _sequentials.append(block(
                                    layer_dict['filters'][i], 
                                    layer_dict['filters'][i + 1], 
                                    3, 
                                    stride=layer_dict['strides'][i],
                                    norm_fn=norm_fn, 
                                    padding=layer_dict['paddings'][i] if len(layer_dict['paddings'][i]) > 1 else layer_dict['paddings'][i][0],
                                    conv_type=layer_dict['types'][i],
                                    indice_key=layer_dict['indice_keys'][i]))
            
            self.downsample_layers.append(spconv.SparseSequential(*_sequentials))                

        last_pad = (0, 0, 0)
        
        out_channels = self.model_cfg.downsample_layers[-1]['filters'][-1] + 2 * 32 * len(self.model_cfg.groupers) * self.repeat_num
        
        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(out_channels, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        
        self.groupers = nn.ModuleList()
        self.grouper_forward_fns = []
        for i in range(len(self.model_cfg.groupers)):
            grouper_dict = self.model_cfg.groupers[i]
            # print(grouper_dict)
            self.groupers.append(grouper_models[grouper_dict['grouper_type']](**grouper_dict.args))
            self.grouper_forward_fns.append(partial(structured_forward_fn[grouper_dict['forward_type']], grouper=self.groupers[-1],
                                            **grouper_dict['maps']))
            
    def forward(self, x, **kwargs):
        rx_list = []
        x = self.conv_input(x)

        for i, layer in enumerate(self.downsample_layers):
            # print(f"layer {i}")
            x = layer(x)
            rx_list.append(x)
        
        lrx_list = []        
        
        for gf_fn in self.grouper_forward_fns:
            _lrx = gf_fn(rx_list, batch_size=kwargs['batch_size'])
            lrx_list.append(_lrx)
        
        lrx = torch.cat([rx_list[-1].features, * lrx_list*self.repeat_num], dim=-1)
        rx_list[-1].features = lrx
        # [200, 176, 5]
        out = self.conv4_out(rx_list[-1])
        
        return out

class BiGNN_reproduce_v1(nn.Module):
    def __init__(self, model_cfg):
        super(BiGNN_reproduce_v1, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.use_voxel_feature = use_voxel_feature

        self.model_cfg = model_cfg
        self.repeat_num = self.model_cfg.repeat_num 
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.model_cfg.conv_inputs[0],  self.model_cfg.conv_inputs[1], 3, padding=1, bias=False, indice_key='subm0'),
            norm_fn(self.model_cfg.conv_inputs[1]),
            nn.ReLU(),
        )
        block = post_act_block
        self.downsample_layers = nn.ModuleList()
        for layer_dict in self.model_cfg.downsample_layers: 
            # for l in layers:
            assert len(layer_dict['types']) == len(layer_dict['indice_keys']), f"{len(layer_dict['types'])} == {len(layer_dict['indice_keys'])}?"
            assert len(layer_dict['types']) == (len(layer_dict['filters'])-1), f"{len(layer_dict['types'])} == {len(layer_dict['filters'])-1}?"
            _sequentials = []
            
            for i in range(len(layer_dict['types'])):
                _sequentials.append(block(
                                    layer_dict['filters'][i], 
                                    layer_dict['filters'][i + 1], 
                                    3, 
                                    stride=layer_dict['strides'][i],
                                    norm_fn=norm_fn, 
                                    padding=layer_dict['paddings'][i] if len(layer_dict['paddings'][i]) > 1 else layer_dict['paddings'][i][0],
                                    conv_type=layer_dict['types'][i],
                                    indice_key=layer_dict['indice_keys'][i]))
            
            self.downsample_layers.append(spconv.SparseSequential(*_sequentials))                

        last_pad = (1, 0, 0)
        
        out_channels = self.model_cfg.downsample_layers[-1]['filters'][-1] + 2 * 32 * len(self.model_cfg.groupers) * self.repeat_num
        
        self.conv4_out = spconv.SparseSequential(
            spconv.SparseConv3d(out_channels, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),  
        )
        
        self.groupers = nn.ModuleList()
        self.grouper_forward_fns = []
        for i in range(len(self.model_cfg.groupers)):
            grouper_dict = self.model_cfg.groupers[i]
            # print(grouper_dict)
            self.groupers.append(grouper_models[grouper_dict['grouper_type']](**grouper_dict.args))
            self.grouper_forward_fns.append(partial(structured_forward_fn[grouper_dict['forward_type']], grouper=self.groupers[-1],
                                            **grouper_dict['maps']))
            
    def forward(self, x, **kwargs):
        rx_list = []
        x = self.conv_input(x)

        for i, layer in enumerate(self.downsample_layers):
            # print(f"layer {i}")
            x = layer(x)
            rx_list.append(x)
        
        lrx_list = []        
        
        for gf_fn in self.grouper_forward_fns:
            _lrx = gf_fn(rx_list, batch_size=kwargs['batch_size'])
            lrx_list.append(_lrx)
        
        lrx = torch.cat([rx_list[-1].features, * lrx_list*self.repeat_num], dim=-1)

        rx_list[-1].features = lrx
        # [200, 176, 5]
        out = self.conv4_out(rx_list[-1])
        
        return out

