from kittidata import KittiLiDAR
def build_dataset(data_cfg):
    return KittiLiDAR(**data_cfg)
