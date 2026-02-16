import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
from typing import Literal
import torch
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int,initialisation: Literal["xavier", "glorot", "he"]="xavier"):
        super().__init__()
        self.dk = embedding_size // n_heads
        self.n_heads = n_heads
        self.w_qkv_h = nn.Linear(in_features=embedding_size,out_features=3*self.dk*n_heads, bias=False)
        self.wo = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=False)
        self.k_cache = None
        self.v_cache = None

        if initialisation=="xavier":
            nn.init.xavier_normal_(self.w_qkv_h.weight)
        elif initialisation=="glorot":
            nn.init.xavier_uniform_(self.w_qkv_h.weight)
        elif initialisation=="he":
            nn.init.kaiming_normal_(self.w_qkv_h.weight)
        else:
            logger.error(f"Initialisation {initialisation} not supported. Please choose between 'xavier','glorot' and 'random'.")
            raise ValueError("Initialisation not supported. Please choose between 'xavier','glorot' and 'random'.")


    def forward(self, x: torch.tensor, mask: torch.tensor = None, use_cache: bool = True):
        # (batch, len, emb size)
        qkv = self.w_qkv_h(x)
        q,k,v = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=self.n_heads)
        
        if use_cache :
            if self.k_cache is not None :
                k = torch.cat((self.k_cache, k), dim=-2)
                v = torch.cat((self.v_cache, v), dim=-2)
            self.k_cache=k.detach()
            self.v_cache=v.detach()
          
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))
        output = F.softmax(scores, dim=-1) @ v
        output = rearrange(output, "b h l d -> b l (h d)", h=self.n_heads)
        return self.wo(output)
