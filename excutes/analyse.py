import numba
import __init__path
import argparse
import torch
import mmcv
import matplotlib as mlp
mlp.use('Agg')
# from mmcv.runner import load_checkpoint, parallel_test
from mmcv.parallel import scatter, collate, MMDataParallel
from dets.tools.evaluation.kitti_eval import get_official_eval_result, get_official_eval_result_v1
from dets.tools.evaluation.coco_utils import results2json, coco_eval
# from dets.datasets import build_dataloader
from dets.datasets.build_dataset import build_dataset
from dets.tools.utils.loader import build_dataloader

# from models import build_detector, detectors
from models.containers.detector import Detector 
# import tools.kitti_common as kitti
import dets.tools.utils.kitti_common as kitti
import numpy as np
# import torch.utils.data
import os
from dets.tools.train_utils import load_params_from_file
from dets.tools.utils import utils
from dets.datasets.build_dataset import build_dataset
import pathlib
import warnings
import logging
from numba import NumbaWarning
from dets.ops.iou3d import iou3d_utils
# with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    # parser.add_argument('--config', default='../configs/cbignn_pswarp_v1.py',  type=str, help='test config file path')
    parser.add_argument('--analyse_dir', default='../analyse', type=str, help='test config file path')
    
    args = parser.parse_args()
    return args


def contrast_difference(gt_dataloader, predicts_dataloaders, iou_fns, iou_thresholds, save_dir, pred_prefixes):
    assert len(predicts_dataloaders) == 2
    for _dataloader in predicts_dataloaders:
        assert len(_dataloader) == len(gt_dataloader)
    gt_template_str = " ".join(['{:.4f}' for _ in pred_prefixes]) + " " + " ".join(['{:.4f}' for _ in pred_prefixes]) + '\n'
    pred_template_str = " ".join(['{:.4f}' for _ in iou_fns]) + '\n'
    data_template = '{} ' + ' '.join(['{:.4f}' for _ in range(15)])

    
    
    data_len = len(gt_dataloader)
    pbar = tqdm.tqdm(total=data_len, leave=False, desc='processing', dynamic_ncols=True)
    gt_dataloader_iter = iter(gt_dataloader)
    dataloader_iters = [iter(_dataloader) for _dataloader in predicts_dataloaders]
    # gt_samples_str = 'gt_instance'
    gt_str = ".".join(pred_prefixes)
    for i in range(data_len):
        gt_data = next(gt_dataloader_iter)
        sample_id = gt_data['img_meta'].data[0][0]['sample_idx'] 
        gt_boxes = gt_data['gt_bboxes'].data[0][0]

        predict_data1 = next(dataloader_iters[0])
        predict_data2 = next(dataloader_iters[1])
        gt_objects =  gt_data['objects'].data[0][0]
        pred1_objects =  predict_data1['objects'].data[0][0]
        pred2_objects =  predict_data2['objects'].data[0][0]


        
        predict_labels1 = predict_data1['gt_labels']
        predict_boxes1 = predict_data1['gt_bboxes'].data[0][0]
        
        predict_labels2 = predict_data2['gt_labels']
        predict_boxes2 = predict_data2['gt_bboxes'].data[0][0]

        predict_1_file = os.path.join(save_dir, pred_prefixes[0], '%06d.txt' % sample_id)
        predict_2_file = os.path.join(save_dir, pred_prefixes[1], '%06d.txt' % sample_id)
        gt_file = os.path.join(save_dir, gt_str, '%06d.txt' % sample_id)

        if gt_boxes is not None:
            gt_ious_3d_1 = iou_fns[0](gt_boxes.cuda(), predict_boxes1.cuda()) if predict_boxes1 is not None else torch.zeros(len(gt_boxes), 1).cuda()
            gt_ious_3d_2 = iou_fns[0](gt_boxes.cuda(), predict_boxes2.cuda()) if predict_boxes2 is not None else torch.zeros(len(gt_boxes), 1).cuda()
            gt_ious_max_3d_1, _ = gt_ious_3d_1.max(dim=1, keepdim=True)
            gt_ious_max_3d_2, _ = gt_ious_3d_2.max(dim=1, keepdim=True)
            gt_ious_max_3ds = torch.cat([gt_ious_max_3d_1, gt_ious_max_3d_2], dim=1)

            gt_ious_bev_1 = iou_fns[1](gt_boxes.cuda(), predict_boxes1.cuda()) if predict_boxes1 is not None else torch.zeros(len(gt_boxes), 1).cuda()
            gt_ious_bev_2 = iou_fns[1](gt_boxes.cuda(), predict_boxes2.cuda()) if predict_boxes2 is not None else torch.zeros(len(gt_boxes), 1).cuda()
            # gt_ious_bev_1 = iou_fns[1](gt_boxes.cuda(), predict_boxes1.cuda()) if predict_boxes1 is not None else torch.zeros(len(gt_boxes), 1).cuda()
            gt_ious_max_bev_1, _ = gt_ious_bev_1.max(dim=1, keepdim=True)
            gt_ious_max_bev_2, _ = gt_ious_bev_2.max(dim=1, keepdim=True)
            gt_ious_max_bevs = torch.cat([gt_ious_max_bev_1, gt_ious_max_bev_2], dim=1)
            with open(gt_file, 'w+') as f:
                for _ious_max_3ds, _ious_max_bevs, obj in zip(gt_ious_max_3ds, gt_ious_max_bevs, gt_objects):
                    iou_line = gt_template_str.format(*_ious_max_3ds, *_ious_max_bevs)
                    data_line = data_template.format(obj.type, obj.truncation, obj.occlusion, obj.alpha, *obj.box2d, obj.h, obj.w, obj.l, *obj.t, obj.ry, obj.score)

                    f.write(data_line + ' ' + iou_line)
        else:
            f = open(gt_file, 'w+')
            f.close()
                
        if predict_boxes1 is not None:
            pred_ious_3d_1 = gt_ious_3d_1 if gt_boxes is not None else torch.zeros(1, len(predict_boxes1)).cuda()
            pred_ious_bev_1 = gt_ious_bev_1 if gt_boxes is not None else torch.zeros(1, len(predict_boxes1)).cuda()
            pred_ious_max_bev_1, _ = pred_ious_bev_1.max(dim=0, keepdim=True)
            pred_ious_max_3d_1, _ = pred_ious_3d_1.max(dim=0, keepdim=True)
            pred_1_ious = torch.cat([pred_ious_max_3d_1, pred_ious_max_bev_1], dim=0).transpose(0, 1)
            with open(predict_1_file, 'w+') as f:
                for _ious_max, obj in zip(pred_1_ious, pred1_objects):
                    iou_line = pred_template_str.format(*_ious_max)
                    data_line = data_template.format(obj.type, obj.truncation, obj.occlusion, obj.alpha, *obj.box2d, obj.h, obj.w, obj.l, *obj.t, obj.ry, obj.score)

                    f.write(data_line + ' ' + iou_line)
        else:
            f = open(predict_1_file, 'w+')
            f.close()
        if predict_boxes2 is not None:
            pred_ious_3d_2 = gt_ious_3d_2 if gt_boxes is not None else torch.zeros(1, len(predict_boxes2)).cuda()
            pred_ious_bev_2 = gt_ious_bev_2 if gt_boxes is not None else torch.zeros(1, len(predict_boxes2)).cuda()
            pred_ious_max_bev_2, _ = pred_ious_bev_2.max(dim=0, keepdim=True)
            pred_ious_max_3d_2, _ = pred_ious_3d_2.max(dim=0, keepdim=True)
            pred_2_ious = torch.cat([pred_ious_max_3d_2, pred_ious_max_bev_2], dim=0).transpose(0, 1)
            with open(predict_2_file, 'w+') as f:
                for _ious_max, obj in zip(pred_2_ious, pred2_objects):
                    iou_line = pred_template_str.format(*_ious_max)
                    data_line = data_template.format(obj.type, obj.truncation, obj.occlusion, obj.alpha, *obj.box2d, obj.h, obj.w, obj.l, *obj.t, obj.ry, obj.score)

                    f.write(data_line + ' ' + iou_line)
        else:
            f = open(predict_2_file, 'w+')
            f.close()
        pbar.update()
        pbar.set_postfix(dict(it=f"{i+1}/{data_len}"))

        
        
def main(args):
    configs = '../configs/analyse_gt.py'
    pred_prefixes = ['cbignn_pswarp_v2', 'cbignn_pswarp_v3']
    contrast_files = ['/chenjiaxin/research/CBi-GNN/experiments/reproduce/cbignn_pswarp_v2/50_p40_docker_env_outs',  '../experiments/reproduce/cbignn_pswarp_v3/50_docker_p40_v3_outs']
    iou_fns = ['RotateIou3dSimilarity', 'RotateIou2dSimilarity']
    iou_fns = [getattr(iou3d_utils, fn)() for fn in  iou_fns]
    gt_root_path = '/chenjiaxin/research/PointRCNN/data/KITTI/object/'
    ann_file = gt_root_path + '../ImageSets/val.txt',
    save_dir = args.analyse_dir
    mmcv.mkdir_or_exist(save_dir)
    mmcv.mkdir_or_exist(os.path.join(save_dir, pred_prefixes[0]))
    mmcv.mkdir_or_exist(os.path.join(save_dir, pred_prefixes[1]))
    gt_str = ".".join(pred_prefixes)
    mmcv.mkdir_or_exist(os.path.join(save_dir, gt_str))

    # os.makedirs()
    iou_threshold = [0.5, 0.7]
    cfg = mmcv.Config.fromfile(configs)
    cfg_name = os.path.splitext(os.path.basename(configs))[0]
    cfg.data.val.with_label = True
    gt_dataset = build_dataset(cfg.data.val)
    gt_dataloader =  build_dataloader(
                    gt_dataset,
                    1,
                    cfg.data.workers_per_gpu,
                    num_gpus=1,
                    shuffle=False,
                    dist=False)
    predicts_datasets = []

    for  i, _dir in enumerate(contrast_files):
        _c = cfg.data.val 
        _c['labels_dir'] = _dir 
        predicts_datasets.append(build_dataset(_c))

    predicts_dataloaders = [build_dataloader(
                            dataset,
                            1,
                            cfg.data.workers_per_gpu,
                            num_gpus=1,
                            shuffle=False,
                            dist=False) for dataset in predicts_datasets]


    contrast_difference(gt_dataloader, predicts_dataloaders, iou_fns, iou_threshold, save_dir, pred_prefixes)

if __name__ == '__main__':
    args = parse_args()
    main(args)
