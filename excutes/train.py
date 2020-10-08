from __future__ import division
import warnings
warnings.filterwarnings('ignore')
# import __init__path
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
print("append %s into system" % os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')))
import argparse
# import sys
# import os
# from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmdet.core import (DistOptimizerHook, CocoDistEvalRecallHook,
                        # CocoDistEvalmAPHook, KittiEvalmAPHook, DistEvalmAPHook)
# from mmdet.datasets import build_dataloader
from dets.datasets.kittidata import KittiLiDAR
from dets.tools.train_utils.envs import get_root_logger, set_random_seed
from dets.tools.train_utils.optimization import build_optimizer, build_scheduler
from dets.tools.train_utils import train_model
import pathlib
from mmcv import Config
from models.containers.detector import Detector 
from dets.datasets.build_dataset import build_dataset
from dets.tools.utils.loader import build_dataloader
from dets.tools.train_utils import load_params_from_file

import warnings
import logging
from numba import NumbaWarning

warnings.filterwarnings("ignore", category=NumbaWarning)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument("config", help='train config file path')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use (only applicable to non-distributed training)')

    # parser.add_argument('config', help='train config file path')
    # parser.add_argument('--work_dir', help='the dir to save logs and models')
    # parser.add_argument(
    #     '--validate',
    #     action='store_true',
    #     help='whether to evaluate the checkpoint during training')
    parser.add_argument('--checkpoint', default=None,  help='checkpoint file')
    # parser.add_argument('--lr', type=float,  help='lr resumed')
    # parser.add_argument('--turns', type=float,  help='number of turn in one epoch')

    # parser.add_argument(
    #     '--gpus',
    #     type=int,
    #     default=1,
    #     help='number of gpus to use '
    #          '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                        default='none',
                        help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_ckpt_save_num', type=int, default=100)
    args = parser.parse_args()
    return args

def main():
    args  = parse_args()
    # print(args.config)
    cfg = Config.fromfile(args.config)   
    pathlib.Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)
    print(f"Experiments are recorded into {cfg.exp_dir}")
    logger = get_root_logger(cfg.exp_dir)
    logger.info(f"Training on {args.gpus} GPUs")
    cfg.gpus= args.gpus
    if args.launcher == 'none':
        distributed = False
    else:
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    
    print("---------------------------------------initialize training members ---------------------------------------")
    model = Detector(cfg.model, cfg.train_cfg, cfg.test_cfg, is_train=True)
    optimizer = build_optimizer(model, cfg.optimizer)
    epoch = 0
    accumulated_iters = 0
    if distributed:
        raise NotImplemented
    else:
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
        if args.checkpoint:
            epoch, accumulated_iters, optimizer_state = load_params_from_file(model, args.checkpoint)        
            optimizer.load_state_dict(optimizer_state)

    # print(model)
    dataset = build_dataset(cfg.data.train)    
    train_loader = build_dataloader(
                    dataset,
                    cfg.data.imgs_per_gpu,
                    cfg.data.workers_per_gpu,
                    dist=distributed)
    
    start_epoch = epoch
    it = accumulated_iters
    last_epoch = -1
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=cfg.total_epochs,
        last_epoch=last_epoch, optim_cfg=cfg.optimizer, lr_cfg=cfg.lr_config
    )
    
    logger.info('---------------------------------Start training---------------------------------')
    train_model(
        model,
        optimizer,
        train_loader,
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.optimizer,
        start_epoch=start_epoch,
        total_epochs=cfg.total_epochs,
        start_iter=it,
        rank=0,
        logger = logger,
        ckpt_save_dir=cfg.exp_dir,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=cfg.checkpoint_config.interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        log_interval = cfg.log_config.interval
    )
    logger.info('---------------------------------End training---------------------------------')
if __name__ == "__main__":
    main()
