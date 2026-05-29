import torch
from torch import nn
# import torch.optim.optimizer
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(Optimizer):
    def __init__(self,params:Iterable[torch.nn.parameter.Parameter],lr:float=1e-3,betas:tuple[float,float]=(0.9,0.95),eps:float=1e-6,weight_decay=0.01):
        # 手写优化器：模型参数信息，学习率，优化器自身参数信息
        defaults={
            'alpha':lr,
            'beta1':betas[0],
            'beta2':betas[1],
            'eps':eps,
            'lamb':weight_decay
        }  # 定义一个参数字典，方便后续索引
        super().__init__(params,defaults)

    def step(self,closure:Callable | None=None):
        loss=None
        if closure is not None:
            closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad=p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                alpha=group['alpha']
                beta_1=group['beta1']
                beta_2=group['beta2']
                eps=group['eps']
                lamba=group['lamb']
                state=self.state[p]
                prev_m=state.get('m',torch.zeros_like(grad))
                state['m']=beta_1*prev_m+(1-beta_1)*grad
                prev_v=state.get('v',torch.zeros_like(grad))
                state['v']=beta_2*prev_v+(1-beta_2)*torch.square(grad)
                t=state.get('t',1)
                alpha_t=alpha*math.sqrt(1-beta_2**t)/(1-beta_1**t)
                p.data-=alpha_t*state['m']/(torch.sqrt(state['v'])+eps)
                p.data-=alpha*lamba*p.data
                state['t']=t+1
        return loss
                


