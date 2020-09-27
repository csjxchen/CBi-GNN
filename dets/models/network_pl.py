import os
from functools import partial
from typing import Optional, Union, Sequence, Dict, Tuple, List

import numpy as np
import pytorch_lightning as pl
from mmcv.parallel import collate
from pytorch_lightning import EvalResult
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dets.dataset.dataset_factory import get_dataset
from tools.test_utils.anno import extract_anno
from tools.train_utils.parse_losses import parse_losses
from .detectors.single_stage import SingleStageDetector

from tools.optimization import OptimWrapper, OneCycle
import tools.kitti_common as kitti


def write_anno(anno, img_idx, saveto):
    template = '{} ' + ' '.join(['{:.4f}' for _ in range(15)]) + '\n'
    if saveto is not None:
        of_path = os.path.join(saveto, '%06d.txt' % img_idx)
        with open(of_path, 'w+') as f:
            for name, bbox, dim, loc, ry, score, alpha in zip(anno['name'], anno["bbox"],
                                                              anno["dimensions"], anno["location"],
                                                              anno["rotation_y"], anno["score"],
                                                              anno["alpha"]):
                line = template.format(name, 0, 0, alpha, *bbox, *dim[[1, 2, 0]], *loc, ry, score)
                f.write(line)


class Cbi_gnn(pl.LightningModule):
    def __init__(self, config):
        super(Cbi_gnn, self).__init__()
        self.cfg = config
        train_cfg, test_cfg = config.train_cfg, config.test_cfg
        self.detectors = SingleStageDetector(train_cfg=train_cfg, test_cfg=test_cfg, **config.model)
        self.optim_cfg = config.optimizer
        self.lr_cfg = config.lr_config
        self.train_dataloader_len = None

    def forward(self, batch):
        return self.detectors(**batch)

    def configure_optimizers(
            self,
    ) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        optim_cfg = self.optim_cfg

        if optim_cfg.type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
        elif optim_cfg.type == 'sgd':
            optimizer = optim.SGD(
                self.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay,
                momentum=optim_cfg.momentum
            )
        elif optim_cfg.type == 'adam_onecycle':
            def children(m: nn.Module):
                return list(m.children())

            def num_children(m: nn.Module) -> int:
                return len(children(m))

            flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
            get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

            optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
            optimizer = OptimWrapper.create(
                optimizer_func, optim_cfg.lr, get_layer_groups(self), wd=optim_cfg.weight_decay, true_wd=True,
                bn_wd=True
            )
        else:
            raise NotImplementedError

        lr_sch = self._configure_lr_sch(optimizer, optim_cfg, self.lr_cfg, total_epochs=self.cfg.total_epochs,
                                        total_iters_each_epoch=self.train_dataloader_len, last_epoch=-1)

        return [optimizer], [lr_sch]

    def train_dataloader(self) -> DataLoader:
        kitti_dataset = get_dataset(self.cfg.data.train)
        batch_size = self.cfg.data.imgs_per_gpu * self.cfg.gpu_count
        num_workers = self.cfg.data.workers_per_gpu * self.cfg.gpu_count
        dataloader = DataLoader(
            dataset=kitti_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=self.cfg.data.imgs_per_gpu),
            pin_memory=False
        )
        self.train_dataloader_len = len(dataloader)
        return dataloader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        kitti_dataset = get_dataset(self.cfg.data.val)
        return DataLoader(
            dataset=kitti_dataset,
            batch_size=1,
            num_workers=self.cfg.data.workers_per_gpu,
            collate_fn=partial(collate, samples_per_gpu=1),
            pin_memory=False

        )

    def training_step(self, batch, batch_idx):
        losses = self(batch)
        loss, log_vars = parse_losses(losses)
        result = pl.TrainResult(minimize=loss)
        # TODO: Anything to log?
        result.log_dict(log_vars)
        return result

    def test_step(self, batch, batch_idx) -> EvalResult:

        results = self(return_loss=False, **batch)

        checkpoint_n = os.path.splitext(os.path.basename(self.cfg.checkpoint))[0]
        if self.cfg.save_to_file:
            saved_dir = os.path.join(self.cfg.work_dir, f"{checkpoint_n}_outs")
        else:
            saved_dir = None
        # with torch.no_grad():
        # results = model(return_loss=False, **data)
        image_shape = (375, 1242)
        annos = []
        for re in results:
            img_idx = re['image_idx']
            if re['bbox'] is not None:
                anno, num_example = extract_anno(image_shape, re, self.cfg.data.val.class_names)
                if num_example != 0:
                    write_anno(anno, img_idx, saved_dir)
                    anno = {n: np.stack(v) for n, v in anno.items()}
                else:
                    anno = kitti.empty_result_anno()
                    if saved_dir is not None:
                        of_path = os.path.join(saved_dir, '%06d.txt' % img_idx)
                        f = open(of_path, 'w+')
                        f.close()
            else:
                anno = kitti.empty_result_anno()
                if saved_dir is not None:
                    of_path = os.path.join(saved_dir, '%06d.txt' % img_idx)
                    f = open(of_path, 'w+')
                    f.close()

            annos.append(anno)

            num_example = annos[-1]["name"].shape[0]
            annos[-1]["image_idx"] = np.array(
                [img_idx] * num_example, dtype=np.int64)
        # TODO: return something meaningful.
        return EvalResult()

    def _configure_lr_sch(self, optimizer, optim_cfg, lr_cfg, total_epochs, total_iters_each_epoch, last_epoch):
        total_steps = total_iters_each_epoch * total_epochs

        if lr_cfg.policy == 'onecycle':
            lr_scheduler = OneCycle(
                optimizer, total_steps, optim_cfg.lr, list(lr_cfg.moms), lr_cfg.div_factor, lr_cfg.pct_start
            )

        elif lr_cfg.policy == 'cosine':
            raise NotImplementedError
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, last_epoch=last_epoch)

        elif lr_cfg.policy == 'step':

            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_cfg.step, last_epoch=last_epoch)

        else:
            raise NotImplementedError

        return lr_scheduler
