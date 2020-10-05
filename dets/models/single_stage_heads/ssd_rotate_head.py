import torch.nn as nn
import numpy as np
from dets.models.utils import one_hot
from mmdet.ops.iou3d import iou3d_utils
from mmdet.ops.iou3d.iou3d_utils import boxes3d_to_bev_torch, boxes_iou_bev
import torch
import torch.nn.functional as F
from mmdet.core.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss, weighted_cross_entropy
from mmdet.core.utils.misc import multi_apply
from mmdet.core.bbox3d.target_ops import create_target_torch
import mmdet.core.bbox3d.box_coders as boxCoders
from mmdet.core.post_processing.bbox_nms import rotate_nms_torch
from functools import partial
# from ..utils import change_default_args, Sequential
# from mmdet.ops.pointnet2.layers_utils import GrouperForGrids, GrouperxyzForGrids


def grid_structured_forward(grids_loc, hrx, batch_size, grouper, hr_voxel_size, offset):
        # lr_indices = lrx.indices.float()
        hr_indices = hrx.indices.float()
        # lr_voxel_size = torch.Tensor(lr_voxel_size).to(lr_indices.device)
        hr_voxel_size = torch.Tensor(hr_voxel_size).to(hr_indices.device)
        offset = torch.Tensor(offset).to(hr_indices.device)
        
        hr_indices[:, 1:] = hr_indices[:, [3, 2, 1]] * hr_voxel_size + \
                        offset + .5 * hr_voxel_size
        
        hr_features = hrx.features
        new_lr_features = []

        for bidx in range(batch_size):
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

def tensor_indices_to_points(hrx, hr_voxel_size, offset):
    # lr_indices = lrx.indices.float()
    hr_indices = hrx.indices.float()
    # lr_voxel_size = torch.Tensor(lr_voxel_size).to(lr_indices.device)
    hr_voxel_size = torch.Tensor(hr_voxel_size).to(hr_indices.device)
    offset = torch.Tensor(offset).to(hr_indices.device)
    
    hr_indices[:, 1:] = hr_indices[:, [3, 2, 1]] * hr_voxel_size + \
                    offset + .5 * hr_voxel_size
    
    hr_features = hrx.features
    
    return hr_indices, hr_features 

def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance,
            decrease other performance.
    """
    # need to convert boxes to z-center format
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = torch.sqrt(la ** 2 + wa ** 2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha  # 1.6

    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return torch.cat([xt, yt, zt, wt, lt, ht, rtx, rty], dim=-1)
    else:
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(
            box_encodings, 1, dim=-1)

    else:
        xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

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

    def forward(self, x):
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        return box_preds, cls_preds, dir_cls_preds

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
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
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
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
                box_preds[opp_labels, -1] += np.pi
            
            bbox_pred = box_preds.view(-1, 7)
            scores = top_scores[selected]

            # bbox_pred = box_preds[selected, :]
            
            if scores.numel() == 0:
                det_bboxes.append(bbox_pred)
                det_scores.append(scores)
                continue
            
            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.extra.nms.iou_thr)

            bbox_pred = bbox_pred[keep, :]
            
            scores = scores[keep]

            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            
            det_scores.append(scores.detach().cpu().numpy())

        return det_bboxes, det_scores
    
    def get_guided_aug_guided_anchors(self, box_preds, cls_preds, dir_cls_preds, anchors, anchors_mask, gt_bboxes, cfg, thr=.1):
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
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
                box_preds[opp_labels, -1] += np.pi
            
            bbox_pred = box_preds.view(-1, 7)
            scores = top_scores[selected]
            # bbox_pred = box_preds[selected, :]
            if scores.numel() == 0:
                det_bboxes.append(bbox_pred)
                det_scores.append(scores)
                continue
            
            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.extra.nms.iou_thr)

            bbox_pred = bbox_pred[keep, :]
            
            scores = scores[keep]

            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            
            det_scores.append(scores.detach().cpu().numpy())

        return det_bboxes, det_scores

def gen_sample_grid(box, window_size=(4, 7), grid_offsets=(0, 0), spatial_scale=1.):
    N = box.shape[0]
    win = window_size[0] * window_size[1]
    xg, yg, wg, lg, rg = torch.split(box, 1, dim=-1)

    xg = xg.unsqueeze_(-1).expand(N, *window_size)
    yg = yg.unsqueeze_(-1).expand(N, *window_size)
    rg = rg.unsqueeze_(-1).expand(N, *window_size)

    cosTheta = torch.cos(rg)
    sinTheta = torch.sin(rg)

    xx = torch.linspace(-.5, .5, window_size[0]).type_as(box).view(1, -1) * wg
    yy = torch.linspace(-.5, .5, window_size[1]).type_as(box).view(1, -1) * lg

    xx = xx.unsqueeze_(-1).expand(N, *window_size)
    yy = yy.unsqueeze_(1).expand(N, *window_size)

    x=(xx * cosTheta + yy * sinTheta + xg)
    y=(yy * cosTheta - xx * sinTheta + yg)

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    return x.view(win, -1), y.view(win, -1)

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W
    
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    
    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1
    # C, 1, N, 1
    return torch.nn.functional.grid_sample(image, samples)

def bilinear_interpolate_torch_gridsample_rep(image, samples_x, samples_y, parts):
    C, H, W = image.shape

    ch = int(C/parts)
    # image = image.unsqueeze(1)  # change to:  C x 1 x H x W
    image = image.view(parts, ch, H, W)
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    
    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between 0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between 0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1
    # C, ch, N, 1
    return torch.nn.functional.grid_sample(image, samples)

def bilinear_interpolate_torch_gridsample_rep_2(image, samples_x, samples_y, parts):
    C, H, W = image.shape
    image = image.unsqueeze(0)
    # ch = int(C/parts)
    # image = image.unsqueeze(1)  # change to:  C x 1 x H x W
    # image = image.view(parts, ch, H, W)
    image = image.expand([parts, C, H, W])
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    
    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between 0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between 0 and 1
    samples = samples * 2 - 1                              # normalize to between -1 and 1
    # C, ch, N, 1
    return torch.nn.functional.grid_sample(image, samples)

def get_hierarchic_mask(boxes, grid_offsets=3.0, spatial_scale=1.):
    
    bottom = boxes[:, 2]
        
    bottom += grid_offsets
        
    bottom_grids = (bottom / spatial_scale).floor()
    
    top_grids = bottom_grids + 2
    
    # print(boxes[:, 2], bottom_grids)
    
    return torch.clamp(bottom_grids, 0, 1), top_grids 

def gen_sample_grid3d_coord(box, window_size=(2, 4, 2), grid_offsets=(0.0, 40.0, 3.0)):
    '''
    '''
    N = box.shape[0]
    win = window_size[0] * window_size[1] * window_size[2]
    xg, yg, zg, wg, lg, hg, rg = torch.split(box, 1, dim=-1)
    
    zg = zg + hg / 2.0
    # print(xg.shape)
    xg = xg.unsqueeze_(-1).unsqueeze_(-1).expand(N, *window_size)
    yg = yg.unsqueeze_(-1).unsqueeze_(-1).expand(N, *window_size)
    zg = zg.unsqueeze_(-1).unsqueeze_(-1).expand(N, *window_size)
    rg = rg.unsqueeze_(-1).unsqueeze_(-1).expand(N, *window_size)

    cosTheta = torch.cos(rg)
    sinTheta = torch.sin(rg)
    
    xx = torch.linspace(-.5, .5, window_size[0]).type_as(box).view(1, -1) * wg
    yy = torch.linspace(-.5, .5, window_size[1]).type_as(box).view(1, -1) * lg
    zz = torch.linspace(-.5, .5, window_size[2]).type_as(box).view(1, -1) * hg
    # print(xx.shape, yy.shape, zz.shape)

    xx = xx.unsqueeze_(-1).unsqueeze_(-1).expand(N, *window_size)
    yy = yy.unsqueeze_(1).unsqueeze_(-1).expand(N, *window_size)
    zz = zz.unsqueeze_(1).unsqueeze_(2).expand(N, *window_size)
    # print(xx.shape, yy.shape, zz.shape)
    
    # zz = torch.linspace(-.5, .5, window_size[2]).type_as(box).view(1, -1) * hg
    x=(xx * cosTheta + yy * sinTheta) + xg
    y=(yy * cosTheta - xx * sinTheta) + yg
    z=(zz + zg)

    x = x.permute(1, 2, 3, 0).contiguous()
    y = y.permute(1, 2, 3, 0).contiguous()
    z = z.permute(1, 2, 3, 0).contiguous()
    

    return x.view(win, -1), y.view(win, -1), z.view(win, -1)

class PSWarpHead(nn.Module):
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

    def forward(self, x, guided_anchors, is_test=False):
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

        if is_test:
            return bbox_scores, guided_anchors
        else:
            return torch.cat(bbox_scores, 0)
    
    def loss(self, cls_preds, gt_bboxes, gt_labels, anchors, cfg):

        batch_size = len(anchors)

        labels, targets, ious = multi_apply(create_target_torch,
                                            anchors, gt_bboxes,
                                            (None,) * batch_size, gt_labels,
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

        cls_loss_reduced = cls_losses / batch_size

        return dict(loss_cls=cls_loss_reduced,)
    
    def get_rescore_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        
        for i in range(len(img_metas)):
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

    def get_guided_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        # similarity_fn = getattr(iou3d_utils, 'RotateIou3dSimilarity')()
        similarity_fn = boxes_iou_bev
        for i in range(len(img_metas)):
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

class PartInteractWarp3(nn.Module):
    def __init__(self, grid_offsets, featmap_stride, in_channels, channels=32, num_class=1, num_parts=49, window_size=(4, 7)):
        super(PartInteractWarp3, self).__init__()
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

    def forward(self, x, guided_anchors, is_test=False):
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
        
        if is_test:
            return bbox_scores, guided_anchors
        else:
            return torch.cat(bbox_scores, 0)
    
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
    
    def loss(self, cls_preds, gt_bboxes, gt_labels, anchors, cfg):
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
    
    def get_rescore_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        
        for i in range(len(img_metas)):
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

    def get_guided_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        # similarity_fn = getattr(iou3d_utils, 'RotateIou3dSimilarity')()
        similarity_fn = boxes_iou_bev
        for i in range(len(img_metas)):
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

class MergeWarp(nn.Module):
    def __init__(self, grid_offsets, featmap_stride, in_channels, channels=32, num_class=1, num_parts=49, window_size=(4, 7)):
        super(MergeWarp, self).__init__()
        
        self._num_class = num_class
        
        self.channels = channels
        
        self.num_parts = num_parts
        
        out_channels = self.channels
        
        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=grid_offsets, window_size=window_size, spatial_scale=1 / featmap_stride)

        self.psconvs = nn.Sequential(
            nn.Conv2d(in_channels, self.num_parts, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_parts, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_parts, self.num_parts, 1, 1, padding=0, bias=False)
        )
        
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
    
    def forward(self, x, guided_anchors, is_test=False):
        nonlocal_x = self.convs(x)
        ps_x = self.psconvs(x)
        # print('x', x.shape)
        bbox_scores = list()
        for i, ga in enumerate(guided_anchors):
            if len(ga) == 0:
                bbox_scores.append(torch.empty(0).type_as(x))
                continue
            
            (xs, ys) = self.gen_grid_fn(ga[:, [0, 1, 3, 4, 6]])
            
            im = nonlocal_x[i]
            
            ps_im = ps_x[i]

            outs = bilinear_interpolate_torch_gridsample_rep_2(im, xs, ys, self.num_parts) #  28, ch, N, 1
            
            ps_outs = bilinear_interpolate_torch_gridsample(ps_im, xs, ys)   #  28, 1, N, 1
            
            # print(ps_outs.shape, outs.shape)
            
            ps_score = torch.mean(ps_outs, 0).view(-1)
            
            nonlocal_score = self.nonlocal_forward(outs) # N, 1, 28, 1
            
            nonlocal_score = torch.mean(nonlocal_score, dim=2).view(-1)

            # print(nonlocal_score.shape, ps_score.shape)
            
            score = (ps_score + nonlocal_score)/2.0

            bbox_scores.append(score)

        if is_test:
            return bbox_scores, guided_anchors
        else:
            return torch.cat(bbox_scores, 0)
    
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
    
    def loss(self, cls_preds, gt_bboxes, gt_labels, anchors, cfg):
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
    
    def get_rescore_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        
        for i in range(len(img_metas)):
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

    def get_guided_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        # similarity_fn = getattr(iou3d_utils, 'RotateIou3dSimilarity')()
        similarity_fn = boxes_iou_bev
        for i in range(len(img_metas)):
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




class PartInteractWarp(nn.Module):
    def __init__(self, grid_offsets, featmap_stride, in_channels, channels=32, num_class=1, num_parts=49, window_size=(4, 7)):
        super(PartInteractWarp, self).__init__()
        
        self._num_class = num_class
        self.channels = channels
        self.num_parts = num_parts
        
        out_channels = num_class * num_parts * self.channels

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

    def forward(self, x, guided_anchors, is_test=False):
        x = self.convs(x)
        # print('x', x.shape)
        bbox_scores = list()
        for i, ga in enumerate(guided_anchors):
            if len(ga) == 0:
                bbox_scores.append(torch.empty(0).type_as(x))
                continue
            (xs, ys) = self.gen_grid_fn(ga[:, [0, 1, 3, 4, 6]])
            im = x[i]
            outs = bilinear_interpolate_torch_gridsample_rep(im, xs, ys, self.num_parts)
            # 28, ch, N, 1
            # print(f"out: {out.shape} len_ga: {len(ga)} xs: {xs.shape} ys: {ys.shape}")
            # score = torch.mean(out, 0).view(-1)
            score = self.nonlocal_forward(outs) # N, 1, 28, 1
            score = torch.mean(score, dim=2).view(-1)
            bbox_scores.append(score)
        
        if is_test:
            return bbox_scores, guided_anchors
        else:
            return torch.cat(bbox_scores, 0)
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
    def loss(self, cls_preds, gt_bboxes, gt_labels, anchors, cfg):
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
    
    def get_rescore_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        
        for i in range(len(img_metas)):
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

    def get_guided_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        # similarity_fn = getattr(iou3d_utils, 'RotateIou3dSimilarity')()
        similarity_fn = boxes_iou_bev
        for i in range(len(img_metas)):
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

class PartInteractWarp2(nn.Module):
    def __init__(self, grid_offsets, featmap_stride, in_channels, channels=32, num_class=1, num_parts=49, window_size=(4, 7)):
        super(PartInteractWarp2, self).__init__()
        self._num_class = num_class
        self.channels = channels
        self.num_parts = num_parts
        out_channels = num_class * num_parts * self.channels
        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=grid_offsets, window_size=window_size, spatial_scale=1 / featmap_stride)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.theta_conv = nn.Sequential(nn.Conv2d(self.channels, int(self.channels/2), 1, 1, padding=0, bias=False), 
                                        nn.BatchNorm2d(int(self.channels/2), eps=1e-3, momentum=0.01),
                                        nn.ReLU(inplace=True)
                                        )
        
        self.phi_conv = nn.Sequential(nn.Conv2d(self.channels, int(self.channels/2), 1, 1, padding=0, bias=False), 
                                        nn.BatchNorm2d(int(self.channels/2), eps=1e-3, momentum=0.01),
                                        nn.ReLU(inplace=True))
        

        self.g_conv = nn.Sequential(nn.Conv2d(self.channels, int(self.channels/2), 1, 1, padding=0, bias=False), 
                                        nn.BatchNorm2d(int(self.channels/2), eps=1e-3, momentum=0.01),
                                        nn.ReLU(inplace=True))
        

        self.out_conv = nn.Sequential(nn.Conv2d(int(self.channels/2), self.channels, 1, 1, padding=0, bias=False), 
                                    nn.BatchNorm2d(self.channels, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))


        self.out_scores = nn.Sequential(
                                    nn.Conv2d(self.channels, self.channels, 1, 1, padding=0, bias=False),
                                    nn.BatchNorm2d(self.channels, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.channels, 1, 1, 1, padding=0, bias=False),
                                    )

    def forward(self, x, guided_anchors, is_test=False):
        x = self.convs(x)
        # print('x', x.shape)
        bbox_scores = list()
        for i, ga in enumerate(guided_anchors):
            if len(ga) == 0:
                bbox_scores.append(torch.empty(0).type_as(x))
                continue
            (xs, ys) = self.gen_grid_fn(ga[:, [0, 1, 3, 4, 6]])
            im = x[i]
            outs = bilinear_interpolate_torch_gridsample_rep(im, xs, ys, self.num_parts)
            # 28, ch, N, 1
            # print(f"out: {out.shape} len_ga: {len(ga)} xs: {xs.shape} ys: {ys.shape}")
            # score = torch.mean(out, 0).view(-1)
            score = self.nonlocal_forward(outs) # N, 1, 28, 1
            score = torch.mean(score, dim=2).view(-1)
            bbox_scores.append(score)
        if is_test:
            return bbox_scores, guided_anchors
        else:
            return torch.cat(bbox_scores, 0)
    
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
    def loss(self, cls_preds, gt_bboxes, gt_labels, anchors, cfg):
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

        cls_loss_reduced = cls_losses / batch_size
        
        return dict(loss_cls=cls_loss_reduced,)
    
    def get_rescore_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        
        for i in range(len(img_metas)):
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
            # bbox_pred




            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            det_scores.append(scores.detach().cpu().numpy())

        return det_bboxes, det_scores

    def get_guided_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()
        # similarity_fn = getattr(iou3d_utils, 'RotateIou3dSimilarity')()
        similarity_fn = boxes_iou_bev
        for i in range(len(img_metas)):
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
