from typing import Any, List
import inspect
import importlib

import torch
import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection

from einops import rearrange

from omegaconf import OmegaConf

from train.utils.utils import get_logger
from train.optim.param_grouping import group_parameters_for_optimizer
from train.utils.checkpoint import load_checkpoint
from train.tasks.seq import SequenceLMModel
# if importlib.util.find_spec('deepspeed'):
#     import deepspeed
#     from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from apex.optimizers import FusedAdam


logger = get_logger(__name__)


class RWKVSequenceLMModel(SequenceLMModel):


    def step(self, batch: Any, is_train=True):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        output = self.forward(x)
        output = rearrange(output, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

    def instantiate_model(self):
        logger.info(f"Instantiating model <{self.model_cfg._target_}>")
        recursive = getattr(self.model_cfg, '_recursive_', False)
        if getattr(self.model_cfg, "_instantiate_config_", True):
            # SE: added the line below to avoid instantiation of custom mixers in the config
            self.args = hydra.utils.instantiate(
                self.model_cfg.config, _recursive_=False, _convert_="object"
            )
            del self.model_cfg.config
            self.model = hydra.utils.instantiate(
                self.model_cfg, _args_=[self.args], _recursive_=False, 
            )
        else:
            self.args = self.model_cfg.config
            del self.model_cfg.config
            self.model = hydra.utils.instantiate(
                self.model_cfg, **self.args, _recursive_=False, 
            )


    @property
    def deepspeed_offload(self) -> bool:
        return False
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False


    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if ("time_mix" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_decay" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

