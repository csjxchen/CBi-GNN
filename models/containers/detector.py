import torch.nn as nn 
from .rfn import RFN
from .rpn import RPN

class Detector(nn.Module):
    def __init__(self, 
                model_cfg,
                train_cfg, 
                test_cfg
                ):
        super(Detector, self).__init__()
        """Container for generalized detector
        Args:
            model_cfg (dict): definite architecture of models
            train_cfg (dict): setting in train_forward
            test_cfg (dict): [description]
        """
        if "rpn" in model_cfg.keys():
            self.rpn  = RPN(model_cfg.rpn, train_cfg.rpn, test_cfg.rpn)

        if "rfn" in model_cfg.keys():
            self.rfn  = RFN(model_cfg.rfn, train_cfg.rfn, test_cfg.rfn)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def forward(self, data_dict, train=True):
        """forward process of detector 
            Args:
                data_dict:{'gt_labels': [class]
                        'gt_boxes': list( [xyzwlhr]) # represented in velodyne coordinates
                        'anchors_mask': list((176 * 200, )    
                        'voxels':  xyz_lidar/voxel_size  # n x 3 FloatTensor, 
                        'coordinates': zyx_indices       # n x 3 FloatTensor, 
                        'num_points':  float32           # num of points in range
                        'img_meta': dict(
                                img_shape=img_shape,
                                sample_idx=sample_id,
                                calib=calib
                                )
                        }   
        """
        # if train:
        data = self.rpn(data_dict)
        if self.rfn is not None:
            data = self.rfn(data_dict) 
        
        
        return data
        