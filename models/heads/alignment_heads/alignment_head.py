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
from abc import ABCMeta, abstractmethod
from models.heads.head_utils import second_box_encode, second_box_decode
from models.utils import change_default_args, Sequential
# from mmdet.ops.pointnet2.layers_utils import GrouperForGrids, GrouperxyzForGrids

class AlignmentHead(nn.Module):
    def __init__(self):
        super(AlignmentHead, self).__init__()
    def forward(self, x, guided_anchors, is_test=False):
        if is_test:
            return self.forward_test(x, guided_anchors)
        else:
            return self.forward_train(x, guided_anchors)
    
    @abstractmethod
    def forward_train(self, x, guided_anchors):
        pass
    
    @abstractmethod
    def forward_test(self, x, guided_anchors):
        pass
    
    def loss(self, cls_preds, anchors, gt_bboxes, gt_labels, cfg):
        batch_size = len(anchors)
        labels, targets, ious = multi_apply(create_target_torch,
                                anchors, gt_bboxes,
                                (None,) * batch_size,
                                gt_labels,
                                similarity_fn=getattr(iou3d_utils, cfg.assigner.similarity_fn)(),
                                box_encoding_fn=second_box_encode,
                                matched_threshold=cfg.assigner.pos_iou_thr,
                                unmatched_threshold=cfg.assigner.neg_iou_thr)

        labels = torch.cat(labels,).unsqueeze_(1)

        # soft_label = torch.clamp(2 * ious - 0.5, 0, 1)
        # labels = soft_label * labels.float()

        cared = labels >= 0
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(torch.float32)
        cls_weights = negative_cls_weights + positives.type(torch.float32)

        pos_normalizer = positives.sum().type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = labels * cared.type_as(labels)
        cls_preds = cls_preds.view(-1, self._num_class)

        cls_losses = weighted_sigmoid_focal_loss(cls_preds, cls_targets.float(), \
                                                 weight=cls_weights, avg_factor=1.)

        cls_loss_reduced = cls_losses * cfg.weight / batch_size
        
        return dict(loss_cls=cls_loss_reduced,)
    
    def get_rescore_bboxes(self, guided_anchors, cls_scores, batch_size, cfg):
        det_bboxes = list()
        det_scores = list()

        for i in range(batch_size):
            bbox_pred = guided_anchors[i]
            # print(bbox_pred.shape[0])
            scores = cls_scores[i]
            if scores.numel == 0:
                det_bboxes.append(None)
                det_scores.append(None)
            bbox_pred = bbox_pred.view(-1, 7)
            scores = torch.sigmoid(scores).view(-1)
            select = scores > cfg.score_thr
            bbox_pred = bbox_pred[select, :]
            scores = scores[select]
            if scores.numel() == 0:
                det_bboxes.append(bbox_pred)
                det_scores.append(scores)
                continue
            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.nms.iou_thr)
            bbox_pred = bbox_pred[keep, :]
            scores = scores[keep]
            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            det_scores.append(scores.detach().cpu().numpy())
        return det_bboxes, det_scores
    
    def get_guided_bboxes(self, guided_anchors, cls_scores, batch_size, cfg):
        det_bboxes = list()
        det_scores = list()
        # similarity_fn = getattr(iou3d_utils, 'RotateIou3dSimilarity')()
        similarity_fn = boxes_iou_bev
        for i in range(batch_size):
            bbox_pred = guided_anchors[i]
            raw_bbox_pred = bbox_pred
            # print(bbox_pred.shape[0])
            scores = cls_scores[i]

            if scores.numel == 0:
                det_bboxes.append(None)
                det_scores.append(None)

            bbox_pred = bbox_pred.view(-1, 7)
            raw_bbox_pred = raw_bbox_pred.view(-1, 7)
            scores = torch.sigmoid(scores).view(-1)
            # select = scores > cfg.score_thr
            # bbox_pred = bbox_pred[select, :]
            # scores = scores[select]
            if scores.numel() == 0:
                det_bboxes.append(bbox_pred)
                det_scores.append(scores)
                continue
            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            # raw_bbox_bev = boxes3d_to_bev_torch(raw_bbox_pred)
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=0.5)

            bbox_pred = bbox_pred[keep, :]
            raw_scores = scores
            # scores = scores[keep]
            # overlaps = similarity_fn(raw_bbox_pred, bbox_pred) # (N, M)
            overlaps = similarity_fn(raw_bbox_pred, bbox_pred) # (N, M)
            merged_preds = []
            for bpi in range(bbox_pred.shape[0]):
                cur_overlaps = overlaps[:, bpi]
                cur_selected = cur_overlaps > 0.7
                cur_selected_preds = raw_bbox_pred[cur_selected]
                cur_selected_raw_scores = raw_scores[cur_selected]
                # cur_selected_scores = 
                cur_selected_preds[:, 6] = cur_selected_preds[:, 6] % (np.pi * 2)
                score_sum = torch.sum(cur_selected_raw_scores)
                normed_sores =  cur_selected_raw_scores/score_sum
                # cur_selected_raw_scores * 
                cur_selected_preds[:, :6] = cur_selected_preds[:, :6] * normed_sores.unsqueeze(-1)
                cur_merge_pred = cur_selected_preds.clone()
                # cur_selected_preds[:, :3] = torch.mean(cur_selected_preds[:, :3], dim=0)
                # cur_selected_preds = torch.mean(cur_selected_preds, dim=0)
                cur_merge_pred[:, :6] = torch.sum(cur_selected_preds[:, :6], dim=0)
                cur_merge_pred = cur_merge_pred.view(-1, 7)
                merged_preds.append(cur_merge_pred)
            merged_preds = torch.cat(merged_preds, dim=0)
            bbox_pred = merged_preds
            # bbox_pred = torch.cat([bbox_pred, merged_preds])
            det_bboxes.append(bbox_pred)

        return det_bboxes