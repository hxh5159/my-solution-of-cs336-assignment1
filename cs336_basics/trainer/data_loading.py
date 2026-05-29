# 此程序的目的是将daraset(数据类型为ndarray)处理成可以输入transformer(LM)中的形式

import torch
import numpy.typing as npt

def data_loading(dataset:npt.NDArray,batch_size:int,context_length:int,device:str) -> tuple[torch.Tensor,torch.Tensor]:
    # context_len:输入句子的最大长度，device:用于处理tensor型数据
    dataset_len=len(dataset)
    inputs=torch.empty(batch_size,context_length,dtype=torch.long)
    targets=torch.empty(batch_size,context_length,dtype=torch.long)
    for i in range(batch_size):
        star_idx=torch.randint(0,dataset_len-context_length,(1, )).item()  # 利用来了torch库中的方法，故生成的数据类型为tensor型，要通过.item()转化为简洁的整数型
        input_seq=dataset[star_idx:star_idx+context_length]
        input_target=dataset[star_idx+1:star_idx+context_length+1]
        inputs[i]=torch.tensor(input_seq,dtype=torch.long)
        targets[i]=torch.tensor(input_target,dtype=torch.long)

    inputs=inputs.to(device)  # 不要忘了将数据转为tensor型后再传入device中
    targets=targets.to(device)
    return (inputs,targets)