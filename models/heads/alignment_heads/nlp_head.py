import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from models.utils import one_hot
from dets.ops.iou3d import iou3d_utils
from dets.ops.iou3d.iou3d_utils import boxes3d_to_bev_torch, boxes_iou_bev
from dets.tools.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss, weighted_cross_entropy
from dets.tools.utils.misc import multi_apply
from dets.tools.bbox3d.target_ops import create_target_torch
import dets.tools.bbox3d.box_coders as boxCoders
from dets.tools.post_processing.bbox_nms import rotate_nms_torch
from functools import partial
# from ..utils import change_default_args, Sequential
# from mmdet.ops.pointnet2.layers_utils import GrouperForGrids, GrouperxyzForGrids
from .alignment_head import AlignmentHead
from models.heads.head_utils import gen_sample_grid
class NonlocalPart(AlignmentHead):
    def __init__(self, grid_offsets, featmap_stride, in_channels, channels=32, num_class=1, num_parts=49, window_size=(4, 7)):
        super(NonlocalPart, self).__init__()
        
        self._num_class = num_class
        self.channels = channels
        self.num_parts = num_parts
        out_channels = self.channels
        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=grid_offsets, window_size=window_size, spatial_scale=1 / featmap_stride)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.theta_conv = nn.Sequential(nn.Conv2d(self.channels, int(self.channels/2), 1, 1, padding=0, bias=False), 
                                        nn.BatchNorm2d(int(self.channels/2), eps=1e-3, momentum=0.01))
        
        self.phi_conv = nn.Sequential(nn.Conv2d(self.channels, int(self.channels/2), 1, 1, padding=0, bias=False), 
                                        nn.BatchNorm2d(int(self.channels/2), eps=1e-3, momentum=0.01))
        
        self.g_conv = nn.Sequential(nn.Conv2d(self.channels, int(self.channels/2), 1, 1, padding=0, bias=False), 
                                        nn.BatchNorm2d(int(self.channels/2), eps=1e-3, momentum=0.01))
        
        self.out_conv = nn.Sequential(nn.Conv2d(int(self.channels/2), self.channels, 1, 1, padding=0, bias=False), 
                                        nn.BatchNorm2d(self.channels, eps=1e-3, momentum=0.01))

        self.out_scores = nn.Sequential(
                                    nn.Conv2d(self.channels, self.channels, 1, 1, padding=0, bias=False),
                                    nn.BatchNorm2d(self.channels, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.channels, 1, 1, 1, padding=0, bias=False),
                                    )

    def forward_train(self, x, guided_anchors):
        x = self.convs(x)
        # print('x', x.shape)
        bbox_scores = list()
        for i, ga in enumerate(guided_anchors):
            if len(ga) == 0:
                bbox_scores.append(torch.empty(0).type_as(x))
                continue
            (xs, ys) = self.gen_grid_fn(ga[:, [0, 1, 3, 4, 6]])
            im = x[i]
            outs = bilinear_interpolate_torch_gridsample_rep_2(im, xs, ys, self.num_parts)
            # 28, ch, N, 1
            score = self.nonlocal_forward(outs) # N, 1, 28, 1
            score = torch.mean(score, dim=2).view(-1)
            bbox_scores.append(score)        
        return torch.cat(bbox_scores, 0)
    
    def forward_test(self, x, guided_anchors):
        x = self.convs(x)
        bbox_scores = list()
        for i, ga in enumerate(guided_anchors):
            if len(ga) == 0:
                bbox_scores.append(torch.empty(0).type_as(x))
                continue
            (xs, ys) = self.gen_grid_fn(ga[:, [0, 1, 3, 4, 6]])
            im = x[i]
            outs = bilinear_interpolate_torch_gridsample_rep_2(im, xs, ys, self.num_parts)
            # 28, ch, N, 1
            score = self.nonlocal_forward(outs) # N, 1, 28, 1
            score = torch.mean(score, dim=2).view(-1)
            bbox_scores.append(score)        
        return bbox_scores, guided_anchors
    
    def nonlocal_forward(self, outs):
        # N, ch, parts, 1
        outs = outs.permute(2, 1, 0, 3)
        # print(outs.shape)
        g_x = self.g_conv(outs)
        g_x = g_x.view(g_x.shape[0], g_x.shape[1], -1).contiguous()
        g_x = g_x.permute(0, 2, 1).contiguous() # N, 28, ch/2
        theta_out = self.theta_conv(outs)
        theta_out = theta_out.view(theta_out.shape[0], theta_out.shape[1], -1).contiguous()
        theta_out = theta_out.permute(0, 2, 1).contiguous() # N, 28, ch/2
        phi_out = self.phi_conv(outs)
        phi_out = phi_out.view(phi_out.shape[0], phi_out.shape[1], -1).contiguous()# N, ch/2, 28
        f = torch.matmul(theta_out, phi_out) # N, 28, 28 
        f_div_c = F.softmax(f, dim=-1) # N, 28, 28
        y = torch.matmul(f_div_c, g_x) # N, 28, ch/2
        y = y.permute(0, 2, 1).contiguous() # N, ch/2, 28
        y = y.unsqueeze(-1) # N, ch/2, parts, 1
        y = self.out_conv(y)
        outs = outs + y
        outs = self.out_scores(outs)
        return outs
    
    
    
    