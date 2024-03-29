from .rpn import RPN
# import models.interfaces as interface 
from models.interfaces import interface_models
from models.necks import neck_models
from models.heads.rpn_heads import rpn_heads_models 
from dets.tools.bbox.transforms import kitti_bbox2results
class CBIGNN(RPN):
    def __init__(self, model_cfg, train_cfg, test_cfg, is_train=True):
        super(CBIGNN, self).__init__(model_cfg, train_cfg, test_cfg, is_train=is_train)    
        self.init_architecture()
        
    def init_architecture(self):
        # initialize backbone
        backbone_dict = self.model_cfg.backbone.copy() if 'backbone' in self.model_cfg.keys() else None
        if backbone_dict:
            backbone_type = backbone_dict.pop('type') 
            self.backbone = interface_models[backbone_type](**backbone_dict)
        else:
            self.backbone = None
        # initialize neck
        neck_dict = self.model_cfg['neck'].copy() if 'neck' in self.model_cfg.keys() else None
        if neck_dict:
            neck_type = neck_dict.pop('type') 
            self.neck = neck_models[neck_type](neck_dict)
        else: 
            self.neck = None
        # initialize head
        rpn_head_dict = self.model_cfg.bbox_head.copy() if "bbox_head" in self.model_cfg.keys() else None
        if rpn_head_dict:
            rpn_head_type = rpn_head_dict.pop('type')
            self.rpn_head = rpn_heads_models[rpn_head_type](**rpn_head_dict.args)
    
    def forward_train(self, data):
        """
        Args:
            data (dict):img, 
                        img_meta, 
                        gt_labels, 
                        gt_bboxes, 
                        voxels, 
                        num_points, 
                        coordinates,
                        anchors
        """
        losses = {}
        
        img_meta = data.pop('img_meta')
        img = data.pop('img')
        batch_size =  len(img_meta)
        data = self.merge_second_batch(data)
        voxel_x = self.backbone(features=data['voxels'])
        neck_outs = self.neck({"voxel_input": voxel_x,  "coords":data['coordinates'], "batch_size": batch_size})
        neck_outs.update(
                {"anchors_mask":data["anchors_mask"],
                    "gt_bboxes": data['gt_bboxes'],
                    "anchors": data['anchors'],
                    "batch_size": batch_size 
                })
        # data [dict]: with keys (rpn_head_features, 
        #                     [align_head_features, 
        #                     anchors_mask,
        #                     gt_bboxes,
        #                     anchors,
        #                     batch_size
        #                     ])
        rpn_outs, alignment_outs = self.rpn_head(neck_outs)
        # achieve loss for rpn
        rpn_loss_inputs = rpn_outs + (data['gt_bboxes'], data['gt_labels'],  data['anchors'], data['anchors_mask'], self.train_cfg) 
        rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
        losses.update(rpn_losses)
        
        if alignment_outs:
            refine_loss_inputs = alignment_outs + (data['gt_bboxes'], data['gt_labels'], self.train_cfg.alignment)
            refine_losses = self.rpn_head.alignment_head.loss(*refine_loss_inputs)
            losses.update(refine_losses)
        return losses
    
    def forward_test(self, data):

        img_meta = data.pop('img_meta')
        
        img = data.pop('img')
        
        batch_size =  len(img_meta)
        
        data = self.merge_second_batch(data)
        
        voxel_x = self.backbone(features=data['voxels'])
        
        neck_outs = self.neck({"voxel_input": voxel_x,  "coords":data['coordinates'], "batch_size": batch_size})

        # prepare data for rpn_head
        neck_outs.update(
                {"anchors_mask":data["anchors_mask"],
                "gt_bboxes": data['gt_bboxes'],
                "anchors":  data['anchors'],
                "batch_size": batch_size,
                'test_alignment_cfg':self.test_cfg.alignment
                })
        
        rpn_outs, alignment_outs = self.rpn_head(neck_outs, is_test=True)
        
        if alignment_outs:
            results = [kitti_bbox2results(*param) for param in zip(*alignment_outs, img_meta)]
        else:
            det_bboxes, det_scores = self.rpn_head.get_guided_dets(*rpn_outs, data['anchors'], data['anchors_mask'], None, self.test_cfg, thr=0.3)
            results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]

        return results