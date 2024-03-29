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

warnings.filterwarnings("ignore", category=NumbaWarning)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
def single_test(model, data_loader, saveto=None, class_names=['Car'], save_ious_file=None):
    template = '{} ' + ' '.join(['{:.4f}' for _ in range(15)]) + '\n'
    if saveto is not None:
        mmcv.mkdir_or_exist(saveto)

    model.eval()
    annos = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    
    for i, data in enumerate(data_loader):
        # print(data.k
        with torch.no_grad():
            results = model(data)
        image_shape = (375,1242)
        for re in results:
            img_idx = re['image_idx']
            if re['bbox'] is not None:
                box2d = re['bbox']
                box3d = re['box3d_camera']
                labels = re['label_preds']
                scores = re['scores']
                alphas = re['alphas']
                anno = kitti.get_start_result_anno()
                num_example = 0
                for bbox2d, bbox3d, label, score, alpha in zip(box2d, box3d, labels, scores, alphas):
                    if bbox2d[0] > image_shape[1] or bbox2d[1] > image_shape[0]:
                        continue
                    if bbox2d[2] < 0 or bbox2d[3] < 0:
                        continue
                    bbox2d[2:] = np.minimum(bbox2d[2:], image_shape[::-1])
                    bbox2d[:2] = np.maximum(bbox2d[:2], [0, 0])
                    anno["name"].append(class_names[int(label)])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    # anno["alpha"].append(-10)
                    anno["alpha"].append(alpha)
                    anno["bbox"].append(bbox2d)
                    # anno["dimensions"].append(np.array([-1,-1,-1]))
                    anno["dimensions"].append(bbox3d[[3, 4, 5]])
                    # anno["location"].append(np.array([-1000,-1000,-1000]))
                    anno["location"].append(bbox3d[:3])
                    # anno["rotation_y"].append(-10)
                    anno["rotation_y"].append(bbox3d[6])
                    anno["score"].append(score)
                    num_example += 1
                
                
                if num_example != 0:
                    if saveto is not None:
                        of_path = os.path.join(saveto, '%06d.txt' % img_idx)
                        with open(of_path, 'w+') as f:
                            for name, bbox, dim, loc, ry, score, alpha in zip(anno['name'], anno["bbox"], \
                            anno["dimensions"], anno["location"], anno["rotation_y"], anno["score"],anno["alpha"]):
                                line = template.format(name, 0, 0, alpha, *bbox, *dim[[1,2,0]], *loc, ry, score)
                                f.write(line)

                    anno = {n: np.stack(v) for n, v in anno.items()}
                    annos.append(anno)
                else:
                    if saveto is not None:
                        of_path = os.path.join(saveto, '%06d.txt' % img_idx)
                        f = open(of_path, 'w+')
                        f.close()
                    annos.append(kitti.empty_result_anno())
            else:
                if saveto is not None:
                    of_path = os.path.join(saveto, '%06d.txt' % img_idx)
                    f = open(of_path, 'w+')
                    f.close()
                annos.append(kitti.empty_result_anno())

            num_example = annos[-1]["name"].shape[0]
            annos[-1]["image_idx"] = np.array(
                [img_idx] * num_example, dtype=np.int64)

        batch_size = len(results)
        for _ in range(batch_size):
            prog_bar.update()

    return annos


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--version', default=None, help='version definition')
    parser.add_argument('--save_to_file', default=False,  help='output result file')
    parser.add_argument('--test', action='store_true',  help='tesing for test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    dataset = build_dataset(cfg.data.val)
    class_names = cfg.data.val.class_names

    
    if args.gpus == 1:
        model = Detector(cfg.model, cfg.train_cfg, cfg.test_cfg, is_train=False)
        model = MMDataParallel(model, device_ids=[0])
        epoch, accumulated_iters, optimizer_state = load_params_from_file(model, args.checkpoint)
        if args.save_to_file:
            # saved_dir = os.path.join(cfg.exp_dir, f"{epoch}_outs")
            saved_dir = os.path.join(cfg.exp_dir, f"{epoch}_outs") if not args.version else  os.path.join(cfg.exp_dir, f"{epoch}_{args.version}_outs")
            print(f"results will be saved into {saved_dir}")
        data_loader = build_dataloader(
            dataset,
            1,
            cfg.data.workers_per_gpu,
            num_gpus=1,
            shuffle=False,
            dist=False)
        outputs = single_test(model, data_loader, saved_dir if args.save_to_file else None, class_names)
    else:
        NotImplementedError
    
    
    if not args.test: 
        # result_file = open(pathlib.Path(cfg.exp_dir) / f'result_{epoch}_outs.txt', 'w')
        result_file_str = f'result_{epoch}_outs.txt' if not args.version  else f'result_{epoch}_{args.version}_outs.txt'
        # result_file = open(pathlib.Path(cfg.exp_dir) / f'result_{epoch}_outs.txt', 'w')
        result_file = open(pathlib.Path(cfg.exp_dir) / result_file_str, 'w')
        # kitti evaluation
        gt_annos = kitti.get_label_annos(dataset.label_prefix, dataset.sample_ids)
        result = get_official_eval_result(gt_annos, outputs, current_classes=class_names)
        print(result)
        print(f"{cfg_name}:{epoch}\n{result}", file=result_file)
        result = get_official_eval_result_v1(gt_annos, outputs, current_class=class_names[0])
        print(result)
        print(result, file=result_file)
        result_file.close()

if __name__ == '__main__':
    main()
