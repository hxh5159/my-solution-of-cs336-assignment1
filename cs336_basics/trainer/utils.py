import torch
from torch import nn
import math
from collections.abc import Iterable

def cross_entropy(out_logit:torch.Tensor,target:torch.Tensor) -> torch.Tensor:
    '''
    out_logit.shape:[batch_size,vocab_size]
    target:这里的target即为下一个token，故其shape为：batch_size
    '''
    get_logit=out_logit.gather(dim=-1,index=target.unsqueeze(-1))
    logsumexp=torch.logsumexp(input=out_logit,dim=-1,keepdim=True)
    loss=-get_logit+logsumexp
    return torch.mean(loss,dim=0,keepdim=False)

def learning_rate_schedule(
        it:int,
        max_learning_rate:float,
        min_learning_rate:float,
        warmup_iters:int,
        cosine_cycle_iters:int,
) -> float:
    assert cosine_cycle_iters>warmup_iters,'无效输入'
    if it<warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif warmup_iters<=it<=cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)*math.pi/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
