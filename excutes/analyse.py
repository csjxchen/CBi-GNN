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

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('analyse_dir', type=str, default='../analyse', help='test config file path')
    
    args = parser.parse_args()
    return args


def contrast_difference(gt_dataloader, predicts_dataloaders, iou_fns, iou_thresholds, save_dir, pred_prefixes):
    assert len(predicts_dataloaders) == 2
    for _dataloader in predicts_dataloaders:
        assert len(_dataloader) == len(gt_dataloader)
    data_len = len(gt_dataloader)
    gt_dataloader_iter = iter(gt_dataloader)
    dataloader_iters = [iter(_dataloader) for _dataloader in predicts_dataloaders]
    gt_samples_str = 'gt_instance'
    for i, gt_data in enumerate(gt_dataloader):
        sample_id = gt_data['img_meta'].data[0][0]['sample_idx'] 
        gt_boxes = gt_data['gt_bboxes'].data[0][0]
        predict_data1 = next(dataloader_iters[0])
        predict_data2 = next(dataloader_iters[1])
        predict_labels1 = predict_data1['gt_labels']
        predict_boxes1 = predict_data1['gt_bboxes'].data[0][0]
        predict_labels2 = predict_data2['gt_labels']
        predict_boxes2 = predict_data2['gt_bboxes'].data[0][0]
        if gt_boxes is not None:
            gt_ious_3d_1 = iou_fns[0](gt_boxes.cuda(), predict_boxes1.cuda()) if predict_boxes1 is not None else torch.zeros(len(gt_boxes), 1)
            gt_ious_3d_2 = iou_fns[0](gt_boxes.cuda(), predict_boxes2.cuda()) if predict_boxes2 is not None else torch.zeros(len(gt_boxes), 1)
            gt_ious_max_3d_1 = gt_ious_3d_1.max(dim=0, keepdim=True)
            gt_ious_max_3d_2 = gt_ious_3d_2.max(dim=0, keepdim=True)
            gt_ious_max_3ds = torch.cat([gt_ious_max_3d_1, gt_ious_max_3d_2], dim=1)



            # gt_ious_bev = iou_fns[0](gt_boxes.cuda(), predict_boxes1.cuda()) if predict_boxes1 is not None else  torch.zeros(len(gt_boxes))
            

        # if gt_boxes is not None or  predict_boxes1 is not None or predict_boxes2 is not None: 
        #     gt_boxes =  gt_boxes.cuda() if gt_boxes is not None else torch.zeros(1, 7).cuda()
        #     predict_boxes1 = predict_boxes1.cuda() if predict_boxes1 is not None else torch.zeros(1, 7).cuda()
        #     predict_boxes2 = predict_boxes2.cuda() if predict_boxes2 is not None else torch.zeros(1, 7).cuda()
        #     ious = iou_fns[0](gt_boxes, predict_boxes1)
        
        # else:
        #     continue
        


def main(argss):
    configs = '../configs/light_cbignn_pswarp_v2.py'
    pred_prefixes = ['cbignn_pswarp_v2', 'cbignn_pswarp_v3']
    contrast_files = ['/chenjiaxin/research/CBi-GNN/experiments/reproduce/cbignn_pswarp_v2/50_p40_docker_env_outs',  '../experiments/reproduce/cbignn_pswarp_v3/50_docker_p40_v3_outs']
    iou_fns = ['RotateIou3dSimilarity', 'RotateIou2dSimilarity']
    iou_fns = [getattr(iou3d_utils, fn)() for fn in  iou_fns]
    gt_root_path = '/chenjiaxin/research/PointRCNN/data/KITTI/object/'
    ann_file = gt_root_path + '../ImageSets/val.txt',
    save_dir = args.analyse_dir
    mmcv.mkdir_or_exist(save_dir)
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


    contrast_difference(gt_dataloader, predicts_dataloaders, iou_fns, iou_threshold)

if __name__ == '__main__':
    main()
