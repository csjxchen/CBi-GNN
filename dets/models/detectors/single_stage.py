import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

from dets.tools.ops.iou3d import iou3d_utils
from mmdet.core import kitti_bbox2results

from .. import builder


class SingleStageDetector(nn.Module):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extra_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
            if hasattr(self.neck, 'use_hier_warp'):
                self.use_hier_warp = self.neck.use_hier_warp
            else:
                self.use_hier_warp = False
        else:
            raise NotImplementedError

        if bbox_head is not None:
            self.rpn_head = builder.build_single_stage_head(bbox_head)

        if extra_head is not None:
            self.extra_head = builder.build_single_stage_head(extra_head)
        else:
            self.extra_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.init_weights(pretrained)

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        assert return_loss == self.training
        if self.training:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    # Handle by lightning
    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = logging.getLogger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)

    def merge_second_batch(self, batch_args):
        ret = {}
        for key, elems in batch_args.items():
            if key in [
                'voxels', 'num_points',
            ]:
                # if key== 'voxels':
                #     print(elems[0].shape)
                ret[key] = torch.cat(elems, dim=0)
            elif key == 'coordinates':
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in [
                'img_meta', 'gt_labels', 'gt_bboxes',
            ]:
                ret[key] = elems
            else:
                ret[key] = torch.stack(elems, dim=0)
        return ret

    def forward_train(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])

        if self.neck.use_hier_warp or self.neck.use_voxel_feature:
            (x, conv6), out3d, point_misc = self.neck(vx, ret['coordinates'], batch_size)
        else:
            (x, conv6), point_misc = self.neck(vx, ret['coordinates'], batch_size)

        # print(f"x {x.shape}, conv6 {conv6.shape}, point_misc {point_misc[0].shape} {point_misc[1].shape} {point_misc[2].shape}")

        losses = dict()
        if self.neck.__class__.__name__ == 'SpMiddleFHD':
            aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
            losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:
            # print(f"for rpn head {x.shape} {conv6.shape}")
            rpn_outs = self.rpn_head(x)
            # print(ret.keys())
            rpn_loss_inputs = rpn_outs + (
                ret['gt_bboxes'], ret['gt_labels'], ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)
            guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'],
                                                              ret['gt_bboxes'], thr=0.1)
            # if len(guided_anchors[])
            max_num_guided_anchors = np.max([len(ga) for ga in guided_anchors])
            # print(num_guided_anchors/len(guided_anchors))
        else:
            raise NotImplementedError

        # bbox head forward and loss
        if self.extra_head:
            if (self.neck.use_hier_warp or self.neck.use_voxel_feature):
                if (max_num_guided_anchors) <= 300:
                    bbox_score = self.extra_head(out3d, guided_anchors)
                    refine_loss_inputs = (
                        bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
                    refine_losses = self.extra_head.loss(*refine_loss_inputs)
                    # print(refine_losses.keys(), rpn_losses.keys())
                    losses.update(refine_losses)
                else:
                    losses.update({'loss_cls': rpn_losses['rpn_cls_loss'] * 0})
            else:
                bbox_score = self.extra_head(conv6, guided_anchors)
                # print(type(ret['gt_bboxes']), ret['gt_bboxes'][0].shape)
                refine_loss_inputs = (
                    bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
                refine_losses = self.extra_head.loss(*refine_loss_inputs)

                losses.update(refine_losses)

        return losses

    def forward_test(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])

        # print(self.neck.use_voxel_feature)
        if self.neck.use_hier_warp or self.neck.use_voxel_feature:
            (x, conv6), out3d = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        else:
            (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        rpn_outs = self.rpn_head.forward(x)
        if not self.extra_head:
            # if True:
            det_bboxes, det_scores = self.rpn_head.get_guided_dets(*rpn_outs, ret['anchors'], ret['anchors_mask'], None,
                                                                   self.test_cfg, thr=0.3)

            results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]

            # _proposals_path = '../ious_with_scores_cls'
            # os.makedirs(_proposals_path, exist_ok=True)
            # for i, res in enumerate(results):
            #     img_idx = res['image_idx']
            #     cur_boxes = det_bboxes[i]
            #     cur_scores = det_scores[i]
            #     cur_gts =  ret['gt_bboxes'][i]
            #     # print(cur_gts)
            #     if cur_boxes.shape[0] == 0 :
            #         continue
            #     _file = os.path.join(_proposals_path, "%06d.txt" % img_idx)
            #     if cur_gts is None:
            #         _ious = torch.zeros(cur_boxes.shape[0])
            #     else:
            #         _ious = self.get_ious(cur_gts.contiguous().cuda(), torch.from_numpy(cur_boxes).contiguous().cuda())
            #     # iou_vs_score = torch.cat([_ious.unsqueeze(1), cur_scores.unsqueeze(1)], dim=1)
            #     template = "{:.4f} {:.4f} \n"
            #     # print(_ious.shape)
            #     with open(_file, 'w+') as f:
            #         for k, _ in enumerate(_ious):
            #             # print(_ious[k],  cur_scores[k])
            #             line = template.format(_ious[k], cur_scores[k])
            #             f.write(line)

            return results

        guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'],
                                                          None, thr=.1)

        # bbox_score, guided_anchors = self.extra_head(conv6, guided_anchors, is_test=True) if (not self.neck.use_hier_warp and not self.neck.use_voxel_feature) else \
        #                             self.extra_head(out3d, guided_anchors, is_test=True)

        # guided_anchors = self.extra_head.get_guided_bboxes(guided_anchors, bbox_score, img_meta,  self.test_cfg.extra)

        bbox_score, guided_anchors = self.extra_head(conv6, guided_anchors, is_test=True) if (
                not self.neck.use_hier_warp and not self.neck.use_voxel_feature) else \
            self.extra_head(out3d, guided_anchors, is_test=True)

        det_bboxes, det_scores = self.extra_head.get_rescore_bboxes(guided_anchors, bbox_score, img_meta,
                                                                    self.test_cfg.extra)

        results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]
        # _proposals_path = '../ious_with_scores_psw'
        # os.makedirs(_proposals_path, exist_ok=True)
        # for i, res in enumerate(results):
        #     img_idx = res['image_idx']
        #     cur_boxes = det_bboxes[i]
        #     cur_scores = det_scores[i]
        #     cur_gts =  ret['gt_bboxes'][i]
        #     # print(cur_gts)
        #     if cur_boxes.shape[0] == 0 :
        #         continue
        #     _file = os.path.join(_proposals_path, "%06d.txt" % img_idx)
        #     if cur_gts is None:
        #         _ious = torch.zeros(cur_boxes.shape[0])
        #     else:
        #         _ious = self.get_ious(cur_gts.contiguous().cuda(), torch.from_numpy(cur_boxes).contiguous().cuda())
        #     # iou_vs_score = torch.cat([_ious.unsqueeze(1), cur_scores.unsqueeze(1)], dim=1)
        #     template = "{:.4f} {:.4f} \n"
        #     # print(_ious.shape)
        #     with open(_file, 'w+') as f:
        #         for k, _ in enumerate(_ious):
        #             # print(_ious[k],  cur_scores[k])
        #             line = template.format(_ious[k], cur_scores[k])
        #             f.write(line)

        # print()
        return results

    def get_mean_ious(self, gt_boxes, boxes, mean_iou=0.0, iou_num=0.0):
        similarity_fn = getattr(iou3d_utils, 'RotateIou3dSimilarity')()
        mean_iou = 0.0
        iou_num = 0
        for bid in range(len(gt_boxes)):
            cur_gts = gt_boxes[bid]
            cur_boxes = boxes[bid]
            ious = similarity_fn(cur_boxes, cur_gts)  # N, M
            _max_ious = ious.max(dim=1)
            _iou_num = cur_boxes.shape[0]
            _mean_iou = _max_ious.sum() / _iou_num if _iou_num > 0 else 0.0
            mean_iou = (mean_iou * iou_num + _mean_iou * _iou_num) / (iou_num + _iou_num)
            iou_num = iou_num + _iou_num
        return mean_iou, iou_num

    def get_ious(self, gt_boxes, boxes):
        similarity_fn = getattr(iou3d_utils, 'RotateIou3dSimilarity')()
        # mean_iou = 0.0
        # iou_num = 0
        # ious = []
        # for bid in  range(len(gt_boxes)):
        _ious = similarity_fn(boxes, gt_boxes)  # N, M
        max_ious, _ = _ious.max(dim=1)
        return max_ious
