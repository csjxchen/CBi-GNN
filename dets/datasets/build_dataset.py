import dets.datasets as datasets 
def build_dataset(data_cfg):
    _data_cfg = data_cfg.copy()
    data_type = _data_cfg.pop('type')

    return getattr(datasets, data_type)(**_data_cfg)