import argparse

import torch
from mmcv import Config
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dets.models.network_pl import Cbi_gnn


def arg_parser():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', type=str, help='train config file path')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument('--work_dir', type=str, default='outputs/CBi', help='the dir to save logs and models')
    # parser.add_argument(
    #     '--validate',
    #     action='store_true',
    #     help='whether to evaluate the checkpoint during training')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    # parser.add_argument('--lr', type=float, help='lr resumed')
    # parser.add_argument('--turns', type=float, help='number of turn in one epoch')

    parser.add_argument(
        '--gpus',
        type=int,
        default=-1,
        help='number of gpus to use, -1 for all gpu available.'
             '(only applicable to non-distributed training)')

    parser.add_argument('--distributed_backend', type=str, default='ddp', help='Backend for distributed module.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--max_ckpt_save_num', type=int, default=100)
    parser.add_argument('--fast_dev_run', action='store_true')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    return args, cfg


def do_train(args, cfg):
    seed_everything(args.seed)
    logger = TensorBoardLogger(save_dir=cfg.work_dir, name='CBi-GNN_logs')

    ckpt_callback = ModelCheckpoint(
        filepath=args.work_dir,  # Modified here to change dir for checkpoint.
        save_top_k=-1,  # -1 for save all checkpoints.
        period=cfg.checkpoint_config.interval
    )

    cfg.gpu_count = get_gpu_count(args)

    cbi_gnn = Cbi_gnn(cfg)

    trainer = Trainer(
        logger=logger,
        checkpoint_callback=ckpt_callback,
        max_epochs=cfg.total_epochs,
        gpus=args.gpus,
        fast_dev_run=args.fast_dev_run,
        distributed_backend=args.distributed_backend
    )
    train_dataloader = cbi_gnn.train_dataloader()
    trainer.fit(
        cbi_gnn,
        train_dataloader=train_dataloader
    )


def get_gpu_count(args):
    if args.gpus == -1:
        return torch.cuda.device_count()
    elif isinstance(args.gpus, int):
        return args.gpus
    else:
        return len(args.gpus)


def do_resume(args, cfg):
    raise NotImplementedError
    # TODO: Add resume.


if __name__ == '__main__':
    args, cfg = arg_parser()
    do_train(args, cfg)
