from .nlp_head import NonlocalPart
from .pswarp import PSWarpHead
alignment_head_models = {"NonlocalPart": NonlocalPart, 
                        'PSWarpHead': PSWarpHead}