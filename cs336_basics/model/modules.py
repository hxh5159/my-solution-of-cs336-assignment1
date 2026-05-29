import torch
import torch.nn as nn
from einops import rearrange, einsum
class Linear(nn.Module):
  def __init__(self,in_features,out_features,device=None,dtype=None):
    super().__init__()
    self.in_features=in_features
    self.out_features=out_features
    self.device=device
    self.dtype=dtype
    self.weight=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))  #先通过torch.empty()创建一个没有初始化的权重矩阵
    self._init_weight()

  def forward(self,x:torch.Tensor) -> torch.Tensor:
    return einsum(x,self.weight,'...d_in,d_out -> ...d_out')

  def _init_weight(self):  # 实现了Xavier初始化
    std=(2/(self.in_features+self.out_features))**0.5
    torch.nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)

class Embedding(nn.Module):  # 这里定义一个Embedding类，将bpe得到的词表中每个词对应的词id转化为词嵌入向量
  def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
    # num_embeddings对应的是vocab_size，一个vocab就对应一个词嵌入向量，embedding_dim对应的是d_model，即词嵌入向量的维数
    super().__init__()
    self.num_embeddings=num_embeddings
    self.embedding_dim=embedding_dim
    self.embed_weight=nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
    self._init_weight()

  def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
    if token_ids.dtype==torch.long:  # 对token_ids的类型进行检查
      pass
    else:
      return self.embed_weight[token_ids]
# self.embed_weight[token_ids]的计算过程：
# token_ids的每个元素作为索引，从embed_weight中取出一行（512维向量）
# 结果形状：[2, 3, 512]

%%writefile /content/my-solution-of-cs336-assignment1/cs336_basics/model/modules.py
import torch
import torch.nn as nn
from einops import rearrange, einsum
class Linear(nn.Module):
  def __init__(self,in_features,out_features,device=None,dtype=None):
    super().__init__()
    self.in_features=in_features
    self.out_features=out_features
    self.device=device
    self.dtype=dtype
    self.weight=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))  #先通过torch.empty()创建一个没有初始化的权重矩阵
    self._init_weight()

  def forward(self,x:torch.Tensor) -> torch.Tensor:
    return einsum(x,self.weight,'...d_in,d_out -> ...d_out')

  def _init_weight(self):  # 实现了Xavier初始化
    std=(2/(self.in_features+self.out_features))**0.5
    torch.nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)

class Embedding(nn.Module):  # 这里定义一个Embedding类，将bpe得到的词表中每个词对应的词id转化为词嵌入向量
  def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
    # num_embeddings对应的是vocab_size，一个vocab就对应一个词嵌入向量，embedding_dim对应的是d_model，即词嵌入向量的维数
    super().__init__()
    self.num_embeddings=num_embeddings
    self.embedding_dim=embedding_dim
    self.embed_weight=nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
    self._init_weight()

  def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
    if token_ids.dtype==torch.long:  # 对token_ids的类型进行检查
      pass
    else:
      return self.embed_weight[token_ids]
# self.embed_weight[token_ids]的计算过程：
# token_ids的每个元素作为索引，从embed_weight中取出一行（512维向量）
# 结果形状：[2, 3, 512]

# 例如：
# token_ids[0,0]=1  → 取出embed_weight[1]（第1个词的512维向量）
# token_ids[0,1]=5  → 取出embed_weight[5]（第5个词的512维向量）
# token_ids[0,2]=3  → 取出embed_weight[3]
# ...依此类推

  def _init_weight(self):
    nn.init.trunc_normal_(self.embed_weight,mean=0,std=1,a=-3,b=3)

class RMSNorm(nn.Module):
  def __init__(self,d_model:int,eps:float=1e-6,device=None,dtype=None):
    super().__init__()
    self.d_model=d_model
    self.eps=eps
    self.g_weight=nn.Parameter(torch.empty(d_model,device=device,dtype=None))
    self._init_weight()
  def forward(self,x:torch.Tensor) -> torch.Tensor:
    in_dtype=x.dtype
    x=x.to(dtype=torch.float32)
    rms=torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
    out=einsum(x/rms,self.g_weight,'...d,d -> ...d')
    return out.to(dtype=in_dtype)

  def _init_weight(self):
    nn.init.trunc_normal_(self.g_weight,mean=0,std=1,a=-3,b=3)

class SwiGLU(nn.Module):

  def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
    super().__init__()
    if dtype is None or not torch.is_floating_point(torch.empty((),dtype=dtype)):  # 这里必须要判别，前面dtype默认设置的都是None
      dtype=torch.float32
    self.model=int(d_model)
    self.d_ff=int(d_ff)
    self.w1=nn.Parameter(torch.empty(self.d_ff,self.d_model,device=device,dtype=dtype))
    self.w3=nn.Parameter(torch.empty(self.d_ff,self.d_model,device=device,dtype=dtype))
    self.w2=nn.Parameter(torch.empty(self.d_model,self.d_ff,device=device,dtype=dtype))
    self._init_weight()

  def _init_weight(self):
    nn.init.trunc_normal_(self.w1,mean=0,std=1,a=-3,b=3)
    nn.init_trunc_normal_(self.w2,mean=0,std=1,a=-3,b=3)
    nn.init_trunc_normal_(self.w3,mean=0,std=1,a=-3,b=3)

  class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,devive=None):
      # 定义query和key的维度为：d_k
      super().__init__()
      self.theta=theta
      self.d_k=d_k
      self.max_seq_len=max_seq_len
      self.freq_arrange=1/(self.theta**(torch.arange(0,self.d_k,2).to(dtype=torch.float)/self.d_k))
      self.register_buffer(name='inv_freq',tensor=self.freq_arrange)

def softmax(x:torch.Tensor,dim:int) -> torch.Tensor:
    x=x-torch.max(x,dim=dim,keepdim=True).values
    x=torch.exp(x)
    return x/torch.sum(x,dim=dim,keepdim=True)

    '''
    在执行点积注意力之前先考虑query和key的维数，注意它们都是对于一个src_sentence的，故shape应该是[batch_size,seq_len,d_k]
    '''
def scaled_dot_product_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor | None=None) -> torch.Tensor:
    # q_k.shape:[batch_size,seq_len,d_k]
    q_k_score=einsum(q,k,'... s_q d,... s_k d -> ... s_q,s_k') / q.size(-1)**0.5
    # 添加掩蔽
    if mask is not None:
      q_k_score=q_k_score.masked_fill(mask==False,float('-inf'))
    q_k_attention=softmax(q_k_score,dim=-1)  # 对q_k_score的最后一维s_k做softmax，得到每个query对所有key的注意力权重，和为1
    return einsum(q_k_attention,v,'... s_q s_k, ... s_k d -> ... s_q d')  # 两个输入和一个输出，一个输入是q_k_attention，注意力权重矩阵，第二个输入v是value，输出是加权求和
    # 这里输出了点积注意力的结果，在q_k_score=einsum(q,k,'... s_q d,... s_k d -> ... s_q,s_k') / q.size(-1)**0.5中已经商掉了根号下d_k

class multihead_self_attention(nn.Module):

  def __init__(self,d_model,num_heads,position_embedding:nn.Module=RotaryPositionalEmbedding,max_seq_len=None,theta=None,token_positions=None,device=None,dtype=None,use_caual_mask=True):
    super().__init__()
    self.pe=None
    self.d_model=d_model
    self.num_heads=num_heads
    self.use_causal_mask=use_causal_mask
    assert d_model%num_heads==0,'num ofheads donen\'match d_model'
    self.d_k=d_model // num_heads
    self.w_q=Linear(self.d_model,self.d_model,device=devide,dtype=dtype)
    self.w_k=Linear(self.d_model,self.d_model,device=device,dtype=dtype)
    self.w_v = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
    self.w_o = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
    if position_embedding is not None and max_seq_len is not None and theta is not None:
      self.pe=position_embedding(theta,self.d_k,max_seq_len)
    self.token_positions=token_positions
  def causal_mask(self,seq_len):
    mask=torch.tril(torch.ones(seq_len,seq_len,device=device,dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # 添加batch_size和num_heads两维

  def forward(self,x:torch.Tensor) -> torch.Tensor:
    # 利用已经得到的线性映射w_q,w_k,w_v将输入的词嵌入矩阵映射为q,k,v
    q_i=self.w_q(x)
    k_i=self.w_k(x)
    v_i=self.w_v(x)
    q_i=rearrange(q_i,'b s (n_h d_k) -> b n_h s d_k',n_h=self.num_heads)
    k_i=rearrange(k_i,'b s (n_h d_k) -> b n_h s d_k',n_h=self.num_heads)
    v_i=rearrange(v_i,'b s (n_h d_k) -> b n_h s d_k',n_h=self.num_heads)
    if self.pe is not None:
      q_i=self.pe(q_i,self.token_positions)
      k_i=self.pe(k_i,self.token_positions)
    mask=None
    if self.use_causal_mask:
      mask=self.causal_mask(q_i.size(-2))
      mask=mask.to(device=q_i.device)
    atten=scaled_dot_product_attention(q_i,k_i,v_i,mask)
    atten=rearrange(atten,'b n_h s d_k -> b s (n_h d_k)',n_h=self.num_heads)

    '''
    在执行点积注意力之前先考虑query和key的维数，注意它们都是对于一个src_sentence的，故shape应该是[batch_size,seq_len,d_k]
    '''
def scaled_dot_product_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor | None=None) -> torch.Tensor:
    # q_k.shape:[batch_size,seq_len,d_k]
    q_k_score=einsum(q,k,'... s_q d,... s_k d -> ... s_q,s_k') / q.size(-1)**0.5
    # 添加掩蔽
    if mask is not None:
      q_k_score=q_k_score.masked_fill(mask==False,float('-inf'))
    q_k_attention=softmax(q_k_score,dim=-1)  # 对q_k_score的最后一维s_k做softmax，得到每个query对所有key的注意力权重，和为1
    return einsum(q_k_attention,v,'... s_q s_k, ... s_k d -> ... s_q d')  # 两个输入和一个输出，一个输入是q_k_attention，注意力权重矩阵，第二个输入v是value，输出是加权求和
    # 这里输出了点积注意力的结果，在q_k_score=einsum(q,k,'... s_q d,... s_k d -> ... s_q,s_k') / q.size(-1)**0.5中已经商掉了根号下d_k

class multihead_self_attention(nn.Module):

  def __init__(self,d_model,num_heads,position_embedding:nn.Module=RotaryPositionalEmbedding,max_seq_len=None,theta=None,token_positions=None,device=None,dtype=None,use_caual_mask=True):
    super().__init__()
    self.pe=None
    self.d_model=d_model
    self.num_heads=num_heads
    self.use_causal_mask=use_causal_mask
    assert d_model%num_heads==0,'num ofheads donen\'match d_model'
    self.d_k=d_model // num_heads
    self.w_q=Linear(self.d_model,self.d_model,device=devide,dtype=dtype)
    self.w_k=Linear(self.d_model,self.d_model,device=device,dtype=dtype)
    self.w_v = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
    self.w_o = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
    if position_embedding is not None and max_seq_len is not None and theta is not None:
      self.pe=position_embedding(theta,self.d_k,max_seq_len)
    self.token_positions=token_positions
  def causal_mask(self,seq_len):
    mask=torch.tril(torch.ones(seq_len,seq_len,device=device,dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # 添加batch_size和num_heads两维

  def forward(self,x:torch.Tensor) -> torch.Tensor:
    # 利用已经得到的线性映射w_q,w_k,w_v将输入的词嵌入矩阵映射为q,k,v
    q_i=self.w_q(x)
    k_i=self.w_k(x)
    v_i=self.w_v(x)
    q_i=rearrange(q_i,'b s (n_h d_k) -> b n_h s d_k',n_h=self.num_heads)
    k_i=rearrange(k_i,'b s (n_h d_k) -> b n_h s d_k',n_h=self.num_heads)
    v_i=rearrange(v_i,'b s (n_h d_k) -> b n_h s d_k',n_h=self.num_heads)
    if self.pe is not None:
      q_i=self.pe(q_i,self.token_positions)
      k_i=self.pe(k_i,self.token_positions)
    mask=None
    if self.use_causal_mask:
      mask=self.causal_mask(q_i.size(-2))
      mask=mask.to(device=q_i.device)
    atten=scaled_dot_product_attention(q_i,k_i,v_i,mask)
    atten=rearrange(atten,'b n_h s d_k -> b s (n_h d_k)',n_h=self.num_heads)
    out=self.w_o(atten)
    return out














