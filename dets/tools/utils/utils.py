import copy
from collections import Sequence
import matplotlib
matplotlib.use('Agg')
import mmcv
from mmcv.runner import obj_from_dict
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from dets import datasets
from dets.tools.point_cloud import voxel_generator
from dets.tools.point_cloud import point_augmentor
from dets.tools.bbox3d import bbox3d_target
from dets.tools.anchor import anchor3d_generator
def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return [to_tensor(d) for d in data]
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    elif data is None:
        return data
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))

def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale

