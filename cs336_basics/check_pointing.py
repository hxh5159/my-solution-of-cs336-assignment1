# 本程序主要实现两个功能：保存checkpoint，重载checkpoint
import torch
import torch.nn

def save_checkpoint(model,optimizer,iteration,out):
    '''
    model:torch.nn.Module
    optimizer:torch.optim.Optimizer
    iteration:int
    out:str
    '''

    torch.save(
        obj={
            # 要保存模型和优化器状态，用字典的形式，还要保存迭代步数，整型
            'model_state':model.state_dict(),
            'optimizer_state':optimizer.state_dict(),
            'iteration':iteration
        },
        f=out
    )

# 重载checkpoint包括：通过字典重载模型状态，优化器状态，返回对应迭代步数
def load_checkpoint(src,model,optimizer) -> torch.long:
    ckp=torch.load(src,map_location='cpu')
    model.load_state_dict(ckp['model_state'])
    optimizer.load_state_dict(ckp['optimizer_state'])
    return ckp['iteration']