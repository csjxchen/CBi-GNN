import os.path as osp
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
import torch
from torch.utils.data import Dataset
from mmcv.runner import obj_from_dict
from mmdet.datasets.transforms import (ImageTransform, BboxTransform)
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.core.bbox3d import bbox3d_target
from mmdet.core.anchor import anchor3d_generator
from mmdet.core.point_cloud import voxel_generator
from mmdet.core.point_cloud import point_augmentor
from mmdet.datasets.kitti_utils import read_label, read_lidar, \
    project_rect_to_velo, Calibration, get_lidar_in_image_fov, \
    project_rect_to_image, project_rect_to_right, load_proposals
from mmdet.core.bbox3d.geometry import rbbox2d_to_near_bbox, filter_gt_box_outside_range, \
    sparse_sum_for_anchors_mask, fused_get_anchors_area, limit_period, center_to_corner_box3d, points_in_rbbox
import os
from mmdet.core.point_cloud.voxel_generator import VoxelGenerator
from mmdet.ops.points_op import points_op_cpu


class KittiLiDAR(Dataset):
    def __init__(self, root,
                 ann_file,
                 img_prefix,
                 img_norm_cfg,
                 img_scale=(1242, 375),
                 size_divisor=32,
                 proposal_file=None,
                 flip_ratio=0.5,
                 with_point=False,
                 with_mask=False,
                 with_label=True,
                 class_names=['Car', 'Van'],
                 augmentor=None,
                 generator=None,
                 anchor_generator=None,
                 anchor_area_threshold=1,
                 target_encoder=None,
                 out_size_factor=2,
                 test_mode=False):
        self.root = root
        # self.img_scales = img_scale if isinstance(img_scale,
        #   list) else [img_scale]
        # assert mmcv.is_list_of(self.img_scales, tuple)

        self.class_names = class_names
        self.test_mode = test_mode
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_point = with_point

        self.lidar_prefix = osp.join(root, 'velodyne_reduced')
        self.calib_prefix = osp.join(root, 'calib')
        self.label_prefix = osp.join(root, 'label_2')

        with open(ann_file, 'r') as f:
            self.sample_ids = list(map(int, f.read().splitlines()))
        self.img_manager = ImageManager(root, img_norm_cfg=img_norm_cfg, size_divisor=size_divisor)
        # self.auxiliary_tools(augmentor, generator, target_encoder, out_size_factor, anchor_area_threshold)

        # def auxiliary_tools(self, augmentor, generator, target_encoder, out_size_factor, anchor_area_threshold):
        # Auxiliary tools.
        self.augmentor = obj_from_dict(augmentor, point_augmentor) if isinstance(augmentor, dict) else augmentor
        self.generator = obj_from_dict(generator, VoxelGenerator) if isinstance(generator, dict) else generator
        # TODO: obj_from_dict from caller.
        self.target_encoder = obj_from_dict(target_encoder, bbox3d_target) if target_encoder is not None else None
        self.out_size_factor = out_size_factor
        self.anchor_area_threshold = anchor_area_threshold
        # anchor
        if anchor_generator is not None:
            feature_map_size = self.generator.grid_size[:2] // self.out_size_factor
            feature_map_size = [*feature_map_size, 1][::-1]
            anchors = anchor_generator(feature_map_size)
            self.anchors = anchors.reshape([-1, 7])
            self.anchors_bv = rbbox2d_to_near_bbox(
                self.anchors[:, [0, 1, 3, 4, 6]])
        else:
            self.anchors = None

    def __len__(self):
        return len(self.sample_ids)

    def _rand_another(self):
        # pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(np.arange(len(self.sample_ids)))

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def prepare_train_img(self, idx):
        sample_id = self.sample_ids[idx]
        # load image
        # img = mmcv.imread(osp.join(self.img_prefix, '%06d.png' % sample_id))
        img, img_shape, pad_shape, scale_factor = self.img_manager(sample_id, scale=1, flip=False)

        # --------------------------------------------------------------------------
        objects = read_label(osp.join(self.label_prefix, '%06d.txt' % sample_id))
        calib = Calibration(osp.join(self.calib_prefix, '%06d.txt' % sample_id))
        gt_bboxes = [obj.box3d for obj in objects if obj.type not in ["DontCare"]]
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_types = [obj.type for obj in objects if obj.type not in ["DontCare"]]
        # --------------------------------------------------------------------------

        # transfer from cam to lidar coordinates
        if len(gt_bboxes) != 0:
            gt_bboxes[:, :3] = project_rect_to_velo(gt_bboxes[:, :3], calib)
            # print(gt_bboxes[:, 2])

        img_meta = dict(
            img_shape=img_shape,
            sample_idx=sample_id,
            calib=calib
        )
        data = dict(
            img=to_tensor(img),
            img_meta=DC(img_meta, cpu_only=True)
        )
        if self.anchors is not None:
            # print("using andchors mask!!!")
            data['anchors'] = DC(to_tensor(self.anchors.astype(np.float32)))

        if self.with_mask:
            NotImplemented

        if self.with_point:
            points = read_lidar(osp.join(self.lidar_prefix, '%06d.bin' % sample_id))

        if self.augmentor is not None and self.test_mode is False:
            sampled_gt_boxes, sampled_gt_types, sampled_points = self.augmentor.sample_all(gt_bboxes, gt_types)
            assert sampled_points.dtype == np.float32
            gt_bboxes = np.concatenate([gt_bboxes, sampled_gt_boxes])
            gt_types = gt_types + sampled_gt_types
            assert len(gt_types) == len(gt_bboxes)

            # to avoid overlapping point (option)
            masks = points_in_rbbox(points, sampled_gt_boxes)
            # masks = points_op_cpu.points_in_bbox3d_np(points[:,:3], sampled_gt_boxes)

            points = points[np.logical_not(masks.any(-1))]

            # paste sampled points to the scene
            points = np.concatenate([sampled_points, points], axis=0)

            # select the interest classes
            selected = [i for i in range(len(gt_types)) if gt_types[i] in self.class_names]
            gt_bboxes = gt_bboxes[selected, :]
            gt_types = [gt_types[i] for i in range(len(gt_types)) if gt_types[i] in self.class_names]

            # force van to have same label as car
            gt_types = ['Car' if n == 'Van' else n for n in gt_types]
            gt_labels = np.array([self.class_names.index(n) + 1 for n in gt_types], dtype=np.int64)

            self.augmentor.noise_per_object_(gt_bboxes, points, num_try=100)
            gt_bboxes, points = self.augmentor.random_flip(gt_bboxes, points)
            gt_bboxes, points = self.augmentor.global_rotation(gt_bboxes, points)
            gt_bboxes, points = self.augmentor.global_scaling(gt_bboxes, points)

        if isinstance(self.generator, VoxelGenerator):
            # voxels, coordinates, num_points = self.generator.generate(points)
            voxel_size = self.generator.voxel_size
            pc_range = self.generator.point_cloud_range
            grid_size = self.generator.grid_size

            keep = points_op_cpu.points_bound_kernel(points, pc_range[:3], pc_range[3:])
            voxels = points[keep, :]
            coordinates = ((voxels[:, [2, 1, 0]] - np.array(pc_range[[2, 1, 0]], dtype=np.float32)) / np.array(
                voxel_size[::-1], dtype=np.float32)).astype(np.int32)
            num_points = np.ones(len(keep)).astype(np.int32)
            # print('voxels', voxels.shape)
            data['voxels'] = DC(to_tensor(voxels.astype(np.float32)))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))

            if self.anchor_area_threshold >= 0 and self.anchors is not None and self.with_mask:
                dense_voxel_map = sparse_sum_for_anchors_mask(
                    coordinates, tuple(grid_size[::-1][1:]))
                dense_voxel_map = dense_voxel_map.cumsum(0)
                dense_voxel_map = dense_voxel_map.cumsum(1)
                anchors_area = fused_get_anchors_area(
                    dense_voxel_map, self.anchors_bv, voxel_size, pc_range, grid_size)
                anchors_mask = anchors_area > self.anchor_area_threshold
                data['anchors_mask'] = DC(to_tensor(anchors_mask.astype(np.bool)))
                # print(data['anchors_mask'].shape, data['anchors_mask'].dtype)
            else:
                N = self.anchors_bv.shape[0]
                anchors_area = np.ones((N), dtype=np.float32) + 10
                anchors_mask = anchors_area > self.anchor_area_threshold
                # print(N, anchors_mask.sum())
                data['anchors_mask'] = DC(to_tensor(anchors_mask.astype(np.bool)))

            # filter gt_bbox out of range
            bv_range = self.generator.point_cloud_range[[0, 1, 3, 4]]
            mask = filter_gt_box_outside_range(gt_bboxes, bv_range)
            gt_bboxes = gt_bboxes[mask]
            gt_labels = gt_labels[mask]

        else:
            NotImplementedError

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # limit rad to [-pi, pi]
        gt_bboxes[:, 6] = limit_period(
            gt_bboxes[:, 6], offset=0.5, period=2 * np.pi)

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
            data['gt_bboxes'] = DC(to_tensor(gt_bboxes))
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        sample_id = self.sample_ids[idx]

        # load image
        # img = mmcv.imread(osp.join(self.img_prefix, '%06d.png' % sample_id))
        img, img_shape, pad_shape, scale_factor = self.img_manager(sample_id, scale=1, flip=False)

        calib = Calibration(osp.join(self.calib_prefix, '%06d.txt' % sample_id))

        if self.with_label:
            objects = read_label(osp.join(self.label_prefix, '%06d.txt' % sample_id))
            gt_bboxes = [obj.box3d for obj in objects if obj.type in self.class_names]

            if len(gt_bboxes) != 0:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.ones(len(gt_bboxes), dtype=np.int64)
                # transfer from cam to lidar coordinates
                gt_bboxes[:, :3] = project_rect_to_velo(gt_bboxes[:, :3], calib)
            else:
                gt_bboxes = None
                gt_labels = None

        img_meta = dict(
            img_shape=img_shape,
            sample_idx=sample_id,
            calib=calib
        )

        data = dict(
            img=to_tensor(img),
            img_meta=DC(img_meta, cpu_only=True)
        )

        if self.anchors is not None:
            data['anchors'] = DC(to_tensor(self.anchors.astype(np.float32)))

        if self.with_mask:
            NotImplemented

        if self.with_point:
            points = read_lidar(osp.join(self.lidar_prefix, '%06d.bin' % sample_id))

        if isinstance(self.generator, VoxelGenerator):
            voxel_size = self.generator.voxel_size
            pc_range = self.generator.point_cloud_range
            grid_size = self.generator.grid_size

            keep = points_op_cpu.points_bound_kernel(points, pc_range[:3], pc_range[3:])
            voxels = points[keep, :]
            coordinates = ((voxels[:, [2, 1, 0]] - np.array(pc_range[[2, 1, 0]], dtype=np.float32)) / np.array(
                voxel_size[::-1], dtype=np.float32)).astype(np.int32)
            num_points = np.ones(len(keep)).astype(np.int32)

            data['voxels'] = DC(to_tensor(voxels.astype(np.float32)))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))

            if self.anchor_area_threshold >= 0 and self.anchors is not None and self.with_mask:
                dense_voxel_map = sparse_sum_for_anchors_mask(
                    coordinates, tuple(grid_size[::-1][1:]))
                dense_voxel_map = dense_voxel_map.cumsum(0)
                dense_voxel_map = dense_voxel_map.cumsum(1)
                anchors_area = fused_get_anchors_area(
                    dense_voxel_map, self.anchors_bv, voxel_size, pc_range, grid_size)
                anchors_mask = anchors_area > self.anchor_area_threshold
                data['anchors_mask'] = DC(to_tensor(anchors_mask.astype(np.bool)))
            else:
                N = self.anchors_bv.shape[0]
                anchors_area = np.ones((N), dtype=np.float32) + 10
                anchors_mask = anchors_area > self.anchor_area_threshold
                # print(N, anchors_mask.sum())
                data['anchors_mask'] = DC(to_tensor(anchors_mask.astype(np.bool)))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels), cpu_only=True)
            data['gt_bboxes'] = DC(to_tensor(gt_bboxes), cpu_only=True)
        else:
            data['gt_labels'] = DC(None, cpu_only=True)
            data['gt_bboxes'] = DC(None, cpu_only=True)

        return data


class ImageManager(object):
    def __init__(self, root, img_norm_cfg, size_divisor=None):
        self.path = osp.join(root, 'image_2')
        # check the size_divisor's effective
        self.transformer = ImageTransform(
            size_divisor=size_divisor, **img_norm_cfg)

    def __call__(self, sample_id, scale, flip=False):
        img = mmcv.imread(osp.join(self.path, '%06d.png' % sample_id))
        img, img_shape, pad_shape, scale_factor = self.transformer(img, scale=1, flip=False)
        return img, img_shape, pad_shape, scale_factor
