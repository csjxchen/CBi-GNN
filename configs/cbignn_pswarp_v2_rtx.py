model = dict(
        rpn=dict(
            type="CBIGNN",
            backbone=dict(
                type='SimpleVoxel',
                num_input_features=4,
                use_norm=True,
                num_filters=[32, 64],
                with_distance=False),
            neck=dict(
                type='CBiNet',
                output_shape=[40, 1600, 1408],
                num_input_features=4,
                num_hidden_features=64 * 4,
                ThrDNet=dict(
                    type="BiGNN",
                    args=dict(
                        conv_inputs=[4, 16],
                        downsample_layers=[{'types':['subm'], 'indice_keys': ['subm1'],  'paddings': [[1]], 'strides':[1],  'filters': [16, 16]},
                                    {'types':['spconv', 'subm', 'subm'], 'indice_keys': ['spconv2', 'subm2', 'subm2'], 'paddings': [[1], [1], [1]],   'strides':[2, 1, 1], 'filters': [16, 32, 32, 32]}, 
                                    {'types':['spconv', 'subm', 'subm'], 'indice_keys': ['spconv3', 'subm3', 'subm3'], 'paddings': [[1], [1], [1]], 'strides':[2, 1, 1], 'filters': [32, 64, 64, 64]}, 
                                    {'types':['spconv', 'subm', 'subm'], 'indice_keys': ['spconv4', 'subm4', 'subm4'], 'paddings': [[0, 1, 1], [1], [1]], 'strides':[2, 1, 1], 'filters': [64, 64, 64, 64]}],
                        groupers=[
                                dict(
                                    grouper_type='GrouperDisAttention_reproduce_v2',
                                    forward_type='v1',
                                    args=dict(
                                        radius=1.0,
                                        nsamples=128,
                                        mlps=[16, 32],
                                        use_xyz=True,
                                        # xyz_mlp_spec=[3, 32, 32],
                                        xyz_mlp_spec=[3, 32],

                                        xyz_mlp_bn=False,
                                        feat_mlp_bn=False,
                                        instance_norm=False
                                        ),
                                    maps=dict(
                                        lr_index=3,
                                        hr_index=0,
                                        lr_voxel_size=(0.4, 0.4, 1.0),   
                                        hr_voxel_size=(0.05, 0.05, 0.1),
                                        offset=(0., -40., -3.)
                                        )),
                                dict(
                                    grouper_type='GrouperDisAttention_reproduce_v2',
                                    forward_type='v1',
                                    args=dict(
                                        radius=1.0,
                                        nsamples=32,
                                        mlps=[32, 32],
                                        use_xyz=True,
                                        # xyz_mlp_spec=[3, 32, 32],
                                        xyz_mlp_spec=[3, 32],
                                        xyz_mlp_bn=False,
                                        feat_mlp_bn=False,
                                        instance_norm=False
                                        ),
                                    maps=dict(
                                        lr_index=3,
                                        hr_index=1,
                                        lr_voxel_size=(0.4, 0.4, 1.0),   
                                        hr_voxel_size=(0.1, 0.1, 0.2),
                                        offset=(0., -40., -3.)
                                        )),
                                dict(
                                    grouper_type='GrouperDisAttention_reproduce_v2',
                                    forward_type='v1',
                                    args=dict(
                                        radius=1.0,
                                        nsamples=16,
                                        mlps=[64, 32],
                                        use_xyz=True,
                                        # xyz_mlp_spec=[3, 32, 32],
                                        xyz_mlp_spec=[3, 32],

                                        xyz_mlp_bn=False,
                                        feat_mlp_bn=False,
                                        instance_norm=False
                                        ),
                                    maps=dict(
                                        lr_index=3,
                                        hr_index=2,
                                        lr_voxel_size=(0.4, 0.4, 1.0),   
                                        hr_voxel_size=(0.2, 0.2, 0.4),
                                        offset=(0., -40., -3.)
                                        )),
                                ])
                            ),
                TwoDNet=dict(
                    type='PCDetBEVNet',
                    args=dict(
                        num_input_features=64 * 4,
                        num_filters=[128, 256],
                        num_output_features=256,
                        concat_input=False, 
                        layer_nums=[5, 5],
                        layer_strides=[1, 2],
                        upsample_strides=[1, 2],
                        num_upsample_filters = [256, 256],
                    )
                )
            ),
            bbox_head=dict(
                type='SSDRotateHead',
                args=dict(
                    num_class=1,
                    num_output_filters=256,
                    num_anchor_per_loc=2,
                    use_sigmoid_cls=True,
                    encode_rad_error_by_sin=True,
                    use_direction_classifier=True,
                    box_code_size=7,
                    alignment_head_cfg=dict(
                        type='PSWarpHead',
                        args=dict(
                            grid_offsets = (0., 40.),
                            featmap_stride=.4,
                            in_channels=256,
                            num_class=1,
                            num_parts=28)
                            )
                    ),
                ),
            ),        
        # rfn=dict()
)

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45, # this one is to limit the force assignment
            ignore_iof_thr=-1,
            similarity_fn ='NearestIouSimilarity'
        ),
        nms=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            nms_thr=0.7,
            min_bbox_size=0
        ),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False,
        alignment=dict(
            weight=1.0,
            assigner=dict(
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1,
                similarity_fn ='RotateIou3dSimilarity')
            )
    )
)

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=100,
        nms_thr=0.7,
        min_bbox_size=0,
        alignment=dict(
            score_thr=0.3, 
            nms=dict(type='nms', iou_thr=0.1), 
            max_per_img=100)
    )
   
)
# dataset settings
dataset_type = 'KittiLiDAR'
data_root = '/chenjiaxin/research/PointRCNN/data/KITTI/object/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root=data_root + 'training/',
        ann_file=data_root + '../ImageSets/train.txt',
        img_prefix=None,
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_label=True,
        with_point=True,
        class_names = ['Car', 'Van'],
        augmentor=dict(
            type='PointAugmentor',
            root_path=data_root,
            info_path=data_root + 'kitti_dbinfos_train.pkl',
            sample_classes=['Car'],
            min_num_points=5,
            sample_max_num=15,
            removed_difficulties=[-1],
            global_rot_range=[-0.78539816, 0.78539816],
            gt_rot_range=[-0.78539816, 0.78539816],
            center_noise_std=[1., 1., .5],
            scale_range=[0.95, 1.05]
        ),
        generator=dict(
            type='VoxelGenerator',
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[0, -40., -3., 70.4, 40., 1.],
            max_num_points=5,
            max_voxels=20000
        ),
        anchor_generator=dict(
            type='AnchorGeneratorStride',
            sizes=[1.6, 3.9, 1.56],
            anchor_strides=[0.4, 0.4, 1.0],
            anchor_offsets=[0.2, -39.8, -1.78],
            rotations=[0, 1.57],
        ),
        anchor_area_threshold=1,
        out_size_factor=8,
        test_mode=False),

    val=dict(
        type=dataset_type,
        root=data_root + 'training/',
        ann_file=data_root + '../ImageSets/val.txt',
        img_prefix=None,
        # img_scale=(1242, 375),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_label=False,
        with_point=True,
        class_names = ['Car'],
        generator=dict(
            type='VoxelGenerator',
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[0., -40., -3., 70.4, 40., 1.],
            max_num_points=5,
            max_voxels=20000
        ),
        anchor_generator=dict(
            type='AnchorGeneratorStride',
            sizes=[1.6, 3.9, 1.56],
            anchor_strides=[0.4, 0.4, 1.0],
            anchor_offsets=[0.2, -39.8, -1.78],
            rotations=[0, 1.57],
        ),
        anchor_area_threshold=1,
        out_size_factor=8,
        test_mode=True),
)
# optimizer
optimizer = dict(
    type='adam_onecycle', lr=0.01, weight_decay=0.001,
    grad_clip=dict(max_norm=10, norm_type=2)
)
# learning policy
lr_config = dict(
    policy='onecycle',
    moms = [0.95, 0.85],
    div_factor = 10,
    pct_start = 0.4
)
checkpoint_config = dict(interval=2)
log_config = dict(interval=50)

total_epochs = 50
dist_params = dict(backend='nccl')
log_level = 'INFO'
exp_dir = '../experiments/reproduce/cbignn_pswarp_v2_rtx'
load_from = None
resume_from = None
workflow = [('train', 1)]
