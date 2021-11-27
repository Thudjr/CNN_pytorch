from torch import optim
import numpy as np


class CosineWarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def createScheduler(optimizer, scheduler, **kwargs):
    if scheduler == None:
        return None
    elif scheduler == 'CosineWarmupLR':
        scheduler = CosineWarmupLR(optimizer, warmup=kwargs['warmup'], max_iters=kwargs['T_max'])
    elif scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=kwargs['step_size'], gamma=kwargs['gamma'])
    elif scheduler == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs['milestones'], gamma=kwargs['gamma'])
    
    return scheduler
  
  
