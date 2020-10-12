import logging
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.runner import get_dist_info

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))

def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError

def _init_dist_slurm(backend, **kwargs):
    raise NotImplementedError

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_root_logger(work_dir):
    filename = '{}.log'.format(time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    log_file = os.path.join(work_dir, filename)
    logging.basicConfig(
        filename=filename,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO)

    logger = logging.getLogger()
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    # file_handler = logging.FileHandler(log_file, 'w')
    # file_handler.setLevel(logging.INFO)
    # logger.addHandler(file_handler)
    # logger.propagate = False
    return logger
