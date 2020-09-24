from functools import partial
from typing import Optional, Union, Sequence, Dict, Tuple, List

import pytorch_lightning as pl
from torch import optim, nn
from torch.optim import Optimizer

from tools.train_utils.parse_losses import parse_losses
from .detectors.single_stage import SingleStageDetector

from tools.optimization import OptimWrapper, OneCycle


class Cbi_gnn(pl.LightningModule):
    def __init__(self, config, train_cfg, test_cfg, train_dataloader_len):
        super(Cbi_gnn, self).__init__()
        self.cfg = config
        self.detectors = SingleStageDetector(train_cfg=train_cfg, test_cfg=test_cfg, **config.model)
        self.optim_cfg = config.optimizer
        self.lr_cfg = config.lr_config
        self.train_dataloader_len = train_dataloader_len

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

        lr_sch = self._configure_lr_sch(optimizer, optim_cfg, self.lr_cfg, total_epochs=self.total_epochs,
                                        total_iters_each_epoch=self.train_dataloader_len, last_epoch=-1)

        return [optimizer], [lr_sch]

    def forward(self, batch):
        return self.detectors(**batch)

    def training_step(self, batch, batch_idx):
        losses = self(batch)
        loss, log_vars = parse_losses(losses)
        result = pl.TrainResult(minimize=loss)
        result.log_dict(log_vars)
        return result

    def _configure_lr_sch(self, optimizer, optim_cfg, lr_cfg, total_epochs, total_iters_each_epoch, last_epoch):
        total_steps = total_iters_each_epoch * total_epochs

        if lr_cfg.policy == 'onecycle':
            lr_scheduler = OneCycle(
                optimizer, total_steps, optim_cfg.lr, list(lr_cfg.moms), lr_cfg.div_factor, lr_cfg.pct_start
            )

        elif lr_cfg.policy == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, last_epoch=last_epoch)

        elif lr_cfg.policy == 'step':

            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_cfg.step, last_epoch=last_epoch)

        else:
            raise NotImplementedError

        return lr_scheduler
