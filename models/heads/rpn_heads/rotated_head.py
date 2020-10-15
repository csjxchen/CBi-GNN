import torch.nn as nn
import numpy as np
from models.utils import one_hot
from dets.ops.iou3d import iou3d_utils
from dets.ops.iou3d.iou3d_utils import boxes3d_to_bev_torch, boxes_iou_bev
import torch
import torch.nn.functional as F
from dets.tools.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss, weighted_cross_entropy
from dets.tools.utils.misc import multi_apply
from dets.tools.bbox3d.target_ops import create_target_torch
from models.heads.head_utils import second_box_decode, second_box_encode
import dets.tools.bbox3d.box_coders as boxCoders
from dets.tools.post_processing.bbox_nms import rotate_nms_torch
from functools import partial
# import models.heads.extra_heads as extra_heads 
from models.heads.alignment_heads import alignment_head_models 

# from ..utils import change_default_args, Sequential
# from mmdet.ops.pointnet2.layers_utils import GrouperForGrids, GrouperxyzForGrids

class SSDRotateHead(nn.Module):
    def __init__(self,
                 num_class=1,
                 num_output_filters=768,
                 num_anchor_per_loc=2,
                 use_sigmoid_cls=True,
                 encode_rad_error_by_sin=True,
                 use_direction_classifier=True,
                 box_coder='GroundBox3dCoder',
                 box_code_size=7,
                 alignment_head_cfg=None
                 ):
        super(SSDRotateHead, self).__init__()
        self._num_class = num_class
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_sigmoid_cls = use_sigmoid_cls
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._use_direction_classifier = use_direction_classifier
        self._box_coder = getattr(boxCoders, box_coder)()
        self._box_code_size = box_code_size
        self._num_output_filters = num_output_filters

        if use_sigmoid_cls:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.conv_cls = nn.Conv2d(num_output_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(
            num_output_filters, num_anchor_per_loc * box_code_size, 1)
        
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                num_output_filters, num_anchor_per_loc * 2, 1)
        # self.extra_head_cfg = extra_head_cfg
        self.alignment_head_cfg = alignment_head_cfg
        self.init_alignment_head()
    
    def init_alignment_head(self):
        if self.alignment_head_cfg:
            _alignment_head_cfg = self.alignment_head_cfg.copy()
            _alignment_head_args = _alignment_head_cfg.args
            self.alignment_head = alignment_head_models[_alignment_head_cfg.type](**_alignment_head_args)
        else:
            self.alignment_head = None
    
    def add_sin_difference(self, boxes1, boxes2):
        rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
                            boxes2[..., -1:])
        rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
        boxes1 = torch.cat((boxes1[..., :-1], rad_pred_encoding), dim=-1)
        boxes2 = torch.cat((boxes2[..., :-1], rad_tg_encoding), dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, anchors, reg_targets, use_one_hot=True):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, 7)
        rot_gt = reg_targets[..., -1] + anchors[..., -1]
        dir_cls_targets = (rot_gt > 0).long()
        if use_one_hot:
            dir_cls_targets = one_hot(
                dir_cls_targets, 2, dtype=anchors.dtype)
        return dir_cls_targets

    def prepare_loss_weights(self, labels,
                             pos_cls_weight=1.0,
                             neg_cls_weight=1.0,
                             loss_norm_type='NormByNumPositives',
                             dtype=torch.float32):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
        reg_weights = positives.type(dtype)
        if loss_norm_type == 'NormByNumExamples':
            num_examples = cared.type(dtype).sum(1, keepdim=True)
            num_examples = torch.clamp(num_examples, min=1.0)
            cls_weights /= num_examples
            bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
        elif loss_norm_type == 'NormByNumPositives':  # for focal loss
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        elif loss_norm_type == 'NormByNumPosNeg':
            pos_neg = torch.stack((positives, negatives), dim=-1).type(dtype)
            normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
            cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
            cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = torch.clamp(normalizer, min=1.0)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer
        else:
            raise ValueError("unknown loss norm type.")
        return cls_weights, reg_weights, cared

    def create_loss(self,
                    box_preds,
                    cls_preds,
                    cls_targets,
                    cls_weights,
                    reg_targets,
                    reg_weights,
                    num_class,
                    use_sigmoid_cls=True,
                    encode_rad_error_by_sin=True,
                    box_code_size=7):
        batch_size = int(box_preds.shape[0])
        box_preds = box_preds.view(batch_size, -1, box_code_size)
        if use_sigmoid_cls:
            cls_preds = cls_preds.view(batch_size, -1, num_class)
        else:
            cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
        one_hot_targets = one_hot(
            cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
        if use_sigmoid_cls:
            one_hot_targets = one_hot_targets[..., 1:]
        if encode_rad_error_by_sin:
            box_preds, reg_targets = self.add_sin_difference(box_preds, reg_targets)

        loc_losses = weighted_smoothl1(box_preds, reg_targets, beta=1 / 9., \
                                       weight=reg_weights[..., None], avg_factor=1.)
        cls_losses = weighted_sigmoid_focal_loss(cls_preds, one_hot_targets, \
                                                 weight=cls_weights[..., None], avg_factor=1.)

        return loc_losses, cls_losses

    def forward(self, data, is_test=False):
        """
        Returns:
            data [dict]: with keys (rpn_head_features, 
                                    [align_head_features, 
                                    anchors_mask,
                                    gt_bboxes,
                                    anchors,
                                    batch_size
                                    ])
        """
        rpn_head_features = data['rpn_head_features']
        box_preds = self.conv_box(rpn_head_features)
        cls_preds = self.conv_cls(rpn_head_features)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(rpn_head_features)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        
        rpn_outs = (box_preds, cls_preds, dir_cls_preds)
        if self.alignment_head_cfg:
            align_head_features = data['align_head_features']
            anchors_mask        = data['anchors_mask']
            gt_bboxes           = data['gt_bboxes']
            anchors             = data['anchors']
            guided_anchors      = self.get_guided_anchors(*rpn_outs, anchors, anchors_mask, gt_bboxes, thr=0.1)
            batch_size          = data['batch_size']
            
            if not is_test:
                bbox_score = self.alignment_head(align_head_features, guided_anchors)
                alignment_outs = (bbox_score, guided_anchors)
                return rpn_outs, alignment_outs
            else:
                bbox_score, guided_anchors = self.alignment_head(align_head_features, guided_anchors, is_test=True)        
                det_bboxes, det_scores = self.alignment_head.get_rescore_bboxes(guided_anchors, bbox_score, batch_size, data['test_alignment_cfg'])
                
                alignment_outs = (det_bboxes, det_scores)
                return rpn_outs, alignment_outs

        return rpn_outs, None

    def loss(self, box_preds, cls_preds, dir_cls_preds, gt_bboxes, gt_labels, anchors, anchors_mask, cfg):

        batch_size = box_preds.shape[0]

        labels, targets, ious = multi_apply(create_target_torch,
                                            anchors, gt_bboxes,
                                            anchors_mask, gt_labels,
                                            similarity_fn=getattr(iou3d_utils, cfg.assigner.similarity_fn)(),
                                            box_encoding_fn = second_box_encode,
                                            matched_threshold=cfg.assigner.pos_iou_thr,
                                            unmatched_threshold=cfg.assigner.neg_iou_thr,
                                            box_code_size=self._box_code_size)


        labels = torch.stack(labels,)
        targets = torch.stack(targets)

        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels)

        cls_targets = labels * cared.type_as(labels)

        loc_loss, cls_loss = self.create_loss(
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights,
            reg_targets=targets,
            reg_weights=reg_weights,
            num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            use_sigmoid_cls=self._use_sigmoid_cls,
            box_code_size=self._box_code_size,
        )

        loc_loss_reduced = loc_loss / batch_size
        loc_loss_reduced *= 2

        cls_loss_reduced = cls_loss / batch_size
        cls_loss_reduced *= 1

        loss = loc_loss_reduced + cls_loss_reduced

        if self._use_direction_classifier:
            dir_labels = self.get_direction_target(anchors, targets, use_one_hot=False).view(-1)
            dir_logits = dir_cls_preds.view(-1, 2)
            weights = (labels > 0).type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = weighted_cross_entropy(dir_logits, dir_labels,
                                              weight=weights.view(-1),
                                              avg_factor=1.)
             
            dir_loss_reduced = dir_loss / batch_size
            dir_loss_reduced *= .2
            loss += dir_loss_reduced

        return dict(rpn_loc_loss=loc_loss_reduced, rpn_cls_loss=cls_loss_reduced, rpn_dir_loss=dir_loss_reduced)

    def get_guided_anchors(self, box_preds, cls_preds, dir_cls_preds, anchors, anchors_mask, gt_bboxes, thr=.1):
        
        batch_size = box_preds.shape[0]
        
        batch_box_preds = box_preds.view(batch_size, -1, self._box_code_size)
        
        batch_anchors_mask = anchors_mask.view(batch_size, -1)
        
        batch_cls_preds = cls_preds.view(batch_size, -1)
        
        batch_box_preds = second_box_decode(batch_box_preds, anchors)

        if self._use_direction_classifier:
            batch_dir_preds = dir_cls_preds.view(batch_size, -1, 2)

        new_boxes = []
        if gt_bboxes is None:
            gt_bboxes = [None] * batch_size

        for box_preds, cls_preds, dir_preds, a_mask, gt_boxes in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_anchors_mask, gt_bboxes
        ):
            # print(a_mask.dtype)
            # print('num', box_preds.shape[0], a_mask.sum())
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]

            if self._use_direction_classifier:
                dir_labels = torch.max(dir_preds, dim=-1)[1]

            if self._use_sigmoid_cls:
                total_scores = torch.sigmoid(cls_preds)
            else:
                total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            top_scores = torch.squeeze(total_scores, -1)

            selected = top_scores > thr

            box_preds = box_preds[selected]

            if self._use_direction_classifier:
                dir_labels = dir_labels[selected]
                # print(dir_labels.dtype)
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                box_preds[opp_labels, -1] += np.pi

            # add ground-truth
            if gt_boxes is not None:
                # print(gt_boxes[:, 2])
                box_preds = torch.cat([gt_boxes, box_preds],0)

            new_boxes.append(box_preds)
        return new_boxes

    def get_guided_dets(self, box_preds, cls_preds, dir_cls_preds, anchors, anchors_mask, gt_bboxes, cfg, thr=.1):
        batch_size = box_preds.shape[0]
        batch_box_preds = box_preds.view(batch_size, -1, self._box_code_size)
        batch_anchors_mask = anchors_mask.view(batch_size, -1)
        batch_cls_preds = cls_preds.view(batch_size, -1)
        batch_box_preds = second_box_decode(batch_box_preds, anchors)

        if self._use_direction_classifier:
            batch_dir_preds = dir_cls_preds.view(batch_size, -1, 2)

        if gt_bboxes is None:
            gt_bboxes = [None] * batch_size
        det_bboxes = list()
        det_scores = list()

        for box_preds, cls_preds, dir_preds, a_mask, gt_boxes in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_anchors_mask, gt_bboxes
        ):
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]

            if self._use_direction_classifier:
                dir_labels = torch.max(dir_preds, dim=-1)[1]

            if self._use_sigmoid_cls:
                total_scores = torch.sigmoid(cls_preds)
            else:
                total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            top_scores = torch.squeeze(total_scores, -1)

            selected = top_scores > thr

            box_preds = box_preds[selected]

            if self._use_direction_classifier:
                dir_labels = dir_labels[selected]
                # print(dir_labels.dtype)
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                box_preds[opp_labels, -1] += np.pi
            
            bbox_pred = box_preds.view(-1, 7)
            scores = top_scores[selected]

            # bbox_pred = box_preds[selected, :]
            
            if scores.numel() == 0:
                det_bboxes.append(bbox_pred)
                det_scores.append(scores)
                continue
            
            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.alignment.nms.iou_thr)

            bbox_pred = bbox_pred[keep, :]
            
            scores = scores[keep]

            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            
            det_scores.append(scores.detach().cpu().numpy())

        return det_bboxes, det_scores