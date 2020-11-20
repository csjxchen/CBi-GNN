
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import torch
import torch.nn.functional as F
class RPN(nn.Module):
    def __init__(self, model_cfg, train_cfg, test_cfg, is_train=True):
        super(RPN, self).__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.is_train = is_train
        # for k, m in self.model_cfg.items():
            # eval_model_str = f"self.{k} = {m.type}(**m)"
            # eval(eval_model_str)
        self.class_names = []
        for k in train_cfg.assigner.keys():
            self.class_names.append(k)
    @abstractmethod
    def init_architecture(self):
        pass
    

    def merge_second_batch(self, batch_args):
        ret = {}
        for key, elems in batch_args.items():
            if key in [
                'voxels', 'num_points',
            ]:
                # if key== 'voxels':
                #     print(elems[0].shape)
                ret[key] = torch.cat(elems, dim=0)
            elif key == 'coordinates':
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in [
                'img_meta', 'gt_labels', 'gt_bboxes', 'gt_types'
            ]:
                ret[key] = elems
            else:
                if isinstance(elems, dict):
                    ret[key] = {k: torch.stack(v, dim=0) for k, v in elems.items()}
                else:
                    ret[key] = torch.stack(elems, dim=0)
        return ret
    # @abstractmethod
    
    def forward(self, data):
        if self.is_train:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    @abstractmethod
    def forward_train(self, data):
        pass
    
    @abstractmethod
    def forward_test(self, data):
        pass