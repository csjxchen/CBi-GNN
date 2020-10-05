import copy

from mmcv.runner import obj_from_dict

import dets.dataset.kittidata as kittidata
from mmdet.core.anchor import anchor3d_generator
from mmdet.core.point_cloud import point_augmentor, voxel_generator
from mmdet.core.bbox3d import bbox3d_target  # TODO: change to our version bbox3d_target


def get_dataset(data_cfg):
    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
        raise NotImplementedError
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    if 'generator' in data_cfg.keys() and data_cfg['generator'] is not None:
        generator = obj_from_dict(data_cfg['generator'], voxel_generator)
    else:
        generator = None

    if 'augmentor' in data_cfg.keys() and data_cfg['augmentor'] is not None:
        augmentor = obj_from_dict(data_cfg['augmentor'], point_augmentor)
    else:
        augmentor = None

    if 'anchor_generator' in data_cfg.keys() and data_cfg['anchor_generator'] is not None:
        anchor_generator = obj_from_dict(data_cfg['anchor_generator'], anchor3d_generator)
    else:
        anchor_generator = None

    if 'target_encoder' in data_cfg.keys() and data_cfg['target_encoder'] is not None:
        target_encoder = obj_from_dict(data_cfg['target_encoder'], bbox3d_target)
    else:
        target_encoder = None

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        if generator is not None:
            data_info['generator'] = generator
        if anchor_generator is not None:
            data_info['anchor_generator'] = anchor_generator
        if augmentor is not None:
            data_info['augmentor'] = augmentor
        if target_encoder is not None:
            data_info['target_encoder'] = target_encoder
        dset = obj_from_dict(data_info, kittidata)  # KittiLiDAR
        dsets.append(dset)
    if len(dsets) > 1:
        # dset = ConcatDataset(dsets)
        raise NotImplementedError
    else:
        dset = dsets[0]
    return dset
