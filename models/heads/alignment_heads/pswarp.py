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
from .alignment_head import AlignmentHead
from models.heads.head_utils import gen_sample_grid, bilinear_interpolate_torch_gridsample
class PSWarpHead(AlignmentHead):
    def __init__(self, grid_offsets, featmap_stride, in_channels, num_class=1, num_parts=49):
        super(PSWarpHead, self).__init__()
        self._num_class = num_class
        out_channels = num_class * num_parts

        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=grid_offsets, spatial_scale=1 / featmap_stride)

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=False)
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
            out = bilinear_interpolate_torch_gridsample(im, xs, ys)
            # print(f"out {out.shape} len_ga {len(ga)}")
            score = torch.mean(out, 0).view(-1)

            bbox_scores.append(score)
        return torch.cat(bbox_scores, 0)
    def forward_test(self, x, guided_anchors):
        x = self.convs(x)
        # print('x', x.shape)
        bbox_scores = list()
        for i, ga in enumerate(guided_anchors):
            if len(ga) == 0:
                bbox_scores.append(torch.empty(0).type_as(x))
                continue
            (xs, ys) = self.gen_grid_fn(ga[:, [0, 1, 3, 4, 6]])
            im = x[i]
            out = bilinear_interpolate_torch_gridsample(im, xs, ys)
            # print(f"out {out.shape} len_ga {len(ga)}")
            score = torch.mean(out, 0).view(-1)

            bbox_scores.append(score)
        return bbox_scores, guided_anchors
    
