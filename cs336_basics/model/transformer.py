import torch
import torch.nn as nn
import cs336_basics.model.modules as modules
from  cs336_basics.model.modules import SwiGLU as FFN

class transformer_block(nn.Module):  # 定义transformer中的单个block
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff

        self.ln1=modules.RMSNorm(d_model)
        self.ln2=modules.RMSNorm(d_model)

        self.attn=modules.multihead_self_attention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta
        )

        # 将rope的inv_freq从state_dict中排除
        if hasattr(self.attn, "pe") and hasattr(self.attn.pe, "inv_freq"):
            buf = self.attn.pe.inv_freq
            
            try:
                
                del self.attn.pe._buffers["inv_freq"]
                self.attn.pe.register_buffer("inv_freq", buf, persistent=False)
            except Exception:
                pass

        self.ffn=FFN(d_model=d_model,d_ff=d_ff)

        if device is not None or dtype is not None:
            self.to(device=device,dtype=dtype)

    def forward(self,in_features:torch.Tensor) -> torch.Tensor:
        '''
        input.shape:[batch_size,seq_len,d_model]
        前向传播计算过程：
        Pre-LN:
        x=x+Atten(LN1(x))
        x=x+FFN(LN2(x))
        '''
        x=in_features
        x=x+self.attn(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x
    
class transformer_lm(nn.Module):  # 利用transformer_block构建基于transformer的语言模型
    def __init__(self,vocab_size:int,context_length:int,num_layers:int,d_model:int,num_heads:int,rope_theta:float,d_ff:int):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.num_layers=num_layers
        self.token_embedding=modules.Embedding(vocab_size,embedding_dim=d_model)  # 将词id序列转化为词嵌入矩阵
        self.layers=nn.ModuleList(
            [
                transformer_block(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    theta=rope_theta
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm=modules.RMSNorm(d_model)
        self.output_embedding=modules.Linear(in_features=d_model,out_features=vocab_size)  # 将transformer输出的d_model线性映射为vocab_size，以便进行softfmax运算
    def forward(self,x:torch.Tensor):
        x=self.token_embedding(x)
        for layer in self.layers:
            x=layer(x)  # 通过transformer的所有blocks
        x=self.output_norm(x)  # 先进行RMSNorm
        x=self.output_embedding(x)  # 将输出的d_model最后映射到vocab_size大小
        return x
