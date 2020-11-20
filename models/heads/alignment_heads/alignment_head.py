import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from models.utils import one_hot
from dets.ops.iou3d import iou3d_utils
from dets.ops.iou3d.iou3d_utils import boxes3d_to_bev_torch, boxes_iou_bev
from dets.tools.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss, weighted_cross_entropy
from dets.tools.utils.misc import multi_apply
from dets.tools.bbox3d.target_ops import create_target_torch_single, create_target_torch_multi
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
        batch_none = (None, ) * batch_size

        # labels, targets, ious = multi_apply(create_target_torch_single,
        #                         anchors, gt_bboxes,
        #                         (None,) * batch_size,
        #                         gt_labels,
        #                         similarity_fn=getattr(iou3d_utils, cfg.assigner.similarity_fn)(),
        #                         box_encoding_fn=second_box_encode,
        #                         matched_threshold=cfg.assigner.pos_iou_thr,
        #                         unmatched_threshold=cfg.assigner.neg_iou_thr)
        labels, targets, ious = multi_apply(create_target_torch_multi,
                                            anchors, batch_none, gt_bboxes, batch_none, batch_none,
                                            similarity_fn=getattr(iou3d_utils, cfg.assigner.similarity_fn)(),
                                            box_encoding_fn = second_box_encode,
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
    
    # def get_rescore_bboxes(self, guided_anchors, cls_scores, batch_size, cfg):
    #     det_bboxes = list()
    #     det_scores = list()

    #     for i in range(batch_size):
    #         bbox_pred = guided_anchors[i]
    #         # print(bbox_pred.shape[0])
    #         scores = cls_scores[i]
    #         if scores.numel == 0:
    #             det_bboxes.append(None)
    #             det_scores.append(None)
    #         bbox_pred = bbox_pred.view(-1, 7)
    #         scores = torch.sigmoid(scores).view(-1)
    #         select = scores > cfg.score_thr
    #         bbox_pred = bbox_pred[select, :]
    #         scores = scores[select]
    #         if scores.numel() == 0:
    #             det_bboxes.append(bbox_pred)
    #             det_scores.append(scores)
    #             continue
    #         boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
    #         keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.nms.iou_thr)
    #         bbox_pred = bbox_pred[keep, :]
    #         scores = scores[keep]
    #         det_bboxes.append(bbox_pred.detach().cpu().numpy())
    #         det_scores.append(scores.detach().cpu().numpy())
    #     return det_bboxes, det_scores
    def get_rescore_bboxes(self, guided_anchors, cls_scores, anchor_labels, batch_size, cfg):
        
        det_bboxes = list()
        det_scores = list()
        det_labels = list()

        for i in range(batch_size):
            bbox_pred = guided_anchors[i]
            scores = cls_scores[i]
            labels = anchor_labels[i]

            if scores.numel() == 0:

                det_bboxes.append(None)
                det_scores.append(None)
                det_labels.append(None)

                continue

            bbox_pred = bbox_pred.view(-1, 7)
            scores = torch.sigmoid(scores).view(-1)
            select = scores > cfg.score_thr

            bbox_pred = bbox_pred[select, :]
            scores = scores[select]
            labels = labels[select]

            if scores.numel() == 0:

                det_bboxes.append(None)
                det_scores.append(None)
                det_labels.append(None)

                continue

            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.nms.iou_thr)

            bbox_pred = bbox_pred[keep, :]
            scores = scores[keep]
            labels = labels[keep]

            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            det_scores.append(scores.detach().cpu().numpy())
            det_labels.append(labels.detach().cpu().numpy())

        return det_bboxes, det_scores, det_labels

    def get_guided_bboxes(self, guided_anchors, cls_scores, anchor_labels, batch_size, cfg):
        det_bboxes = list()
        det_scores = list()
        det_labels = list()

        for i in range(len(img_metas)):
            bbox_pred = guided_anchors[i]
            scores = cls_scores[i]
            labels = anchor_labels[i]

            if scores.numel() == 0:

                det_bboxes.append(None)
                det_scores.append(None)
                det_labels.append(None)

                continue

            bbox_pred = bbox_pred.view(-1, 7)
            scores = torch.sigmoid(scores).view(-1)
            select = scores > cfg.score_thr

            bbox_pred = bbox_pred[select, :]
            scores = scores[select]
            labels = labels[select]

            if scores.numel() == 0:

                det_bboxes.append(None)
                det_scores.append(None)
                det_labels.append(None)

                continue

            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.nms.iou_thr)

            bbox_pred = bbox_pred[keep, :]
            scores = scores[keep]
            labels = labels[keep]

            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            det_scores.append(scores.detach().cpu().numpy())
            det_labels.append(labels.detach().cpu().numpy())
        return det_bboxes, det_scores, det_labels