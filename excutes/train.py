from __future__ import division
# import __init__path
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
print("append %s into system" % os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')))
import argparse
# import sys
# import os
# from mmcv.runner import Runner, DistSamplerSeedHook
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmdet.core import (DistOptimizerHook, CocoDistEvalRecallHook,
                        # CocoDistEvalmAPHook, KittiEvalmAPHook, DistEvalmAPHook)
# from mmdet.datasets import build_dataloader
from dets.datasets.kittidata import KittiLiDAR
from dets.tools.train_utils.envs import get_root_logger, set_random_seed
# from tools.env import get_root_logger, init_dist, set_random_seed
# from tools.train_utils import train_model
import pathlib
from mmcv import Config
from models.containers.detector import Detector 
from dets.datasets.build_dataset import build_dataset
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument("config", help='train config file path')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use (only applicable to non-distributed training)')
    # parser.add_argument("--exp_dir", help='the dir to save logs and models')

    # parser.add_argument('config', help='train config file path')
    # parser.add_argument('--work_dir', help='the dir to save logs and models')
    # parser.add_argument(
    #     '--validate',
    #     action='store_true',
    #     help='whether to evaluate the checkpoint during training')
    # parser.add_argument('--checkpoint', default=None,  help='checkpoint file')
    # parser.add_argument('--lr', type=float,  help='lr resumed')
    # parser.add_argument('--turns', type=float,  help='number of turn in one epoch')

    # parser.add_argument(
    #     '--gpus',
    #     type=int,
    #     default=1,
    #     help='number of gpus to use '
    #          '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--max_ckpt_save_num', type=int, default=100)
    args = parser.parse_args()
    return args

def main():
    args  = parse_args()
    # print(args.config)
    config = Config.fromfile(args.config)   
    pathlib.Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
    print(f"Experiments are recorded into {config.exp_dir}")
    logger = get_root_logger(config.exp_dir)
    logger.info(f"Training on {args.gpus} GPUs")
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    
    model = Detector(config.model, config.train_cfg, config.test_cfg)
    print(model)
    dataset = build_dataset(config.data.train)

if __name__ == "__main__":
    main()
