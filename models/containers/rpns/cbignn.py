from models.containers.rpn import RPN
from models import interface 




class CBIGNN(RPN):
    def __init__(self, model_cfg, train_cfg, test_cfg, is_train=True):
        super(CBIGNN, self).__init__(model_cfg, train_cfg, test_cfg, is_train=True)    
        

    def init_architecture(self):
        # initialize backbone
        backbone_dict = self.model_cfg.backbone.copy() if 'backbone' in self.model_cfg.keys() else None
        if backbone_dict:
            backbone_type = backbone_dict.pop('type') 
            self.backbone = interface[backbone_type][**backbone_dict]
        else:
            self.backbone = None
        
        # initialize neck
        neck_dict = self.model_cfg['neck'].copy() if 'neck' in self.model_cfg.keys() else None
        if neck_dict:
            neck_type = neck_dict.pop('type') 
            self.neck = interface[neck_type][*neck_dict*]
        else: 
            self.neck = None
        
        
        # initialize head
        rpn_head_dict = self.model_cfg.bbox_head.copy() if "bbox_head" in self.model_cfg.keys() else None
        if rpn_head_dict:
            rpn_head_type = rpn_head_dict.pop('type')
            self.rpn_head = 
        
        pass
    # def forward(self, data)