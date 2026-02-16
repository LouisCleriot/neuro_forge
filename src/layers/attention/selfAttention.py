from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SelfAttentionTextBook(nn.Module):
    def __init__(self, embeding_size: int, attention_dim: int, initialisation: Literal["xavier", "glorot", "he"]="xavier") -> None :
        super().__init__()
        
        self.dk = attention_dim
        self.WQ = nn.Linear(in_features=embeding_size, out_features=attention_dim, bias=False)
        self.WK = nn.Linear(in_features=embeding_size, out_features=attention_dim, bias=False)
        self.WV = nn.Linear(in_features=embeding_size, out_features=embeding_size, bias=False)
        self.softmax = nn.Softmax(dim=1)

        if initialisation=="xavier":
            nn.init.xavier_normal_(self.WQ.weight)
            nn.init.xavier_normal_(self.WK.weight)
            nn.init.xavier_normal_(self.WV.weight)
        elif initialisation=="glorot":
            nn.init.xavier_uniform_(self.WQ.weight)
            nn.init.xavier_uniform_(self.WK.weight)
            nn.init.xavier_uniform_(self.WV.weight)
        elif initialisation=="he":
            nn.init.kaiming_normal_(self.WQ.weight)
            nn.init.kaiming_normal_(self.WK.weight)
            nn.init.kaiming_normal_(self.WV.weight)
        else:
            logger.error(f"Initialisation {initialisation} not supported. Please choose between 'xavier','glorot' and 'random'.")
            raise ValueError("Initialisation not supported. Please choose between 'xavier','glorot' and 'random'.")

    def forward(self, x: torch.tensor, mask: torch.tensor = None):
        # x shape: (Batch, Seq_Len, Embedding_Size)

        q = self.WQ(x)   
        k = self.WK(x)
        v = self.WV(x)
        
        scores = torch.matmul(q,k.transpose(-2, -1)) / np.sqrt(self.dk)
        
        if mask is not None: 
            scores = scores.masked_fill(mask==0, float("-inf"))
        attention_weight = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weight, v)
        return output

class SelfAttentionOptimized(nn.Module):
    def __init__(self, embeding_size: int, attention_size: int, initialisation: Literal["xavier", "glorot", "he"]="xavier") -> None :
        super().__init__()
        self.w_qkv = nn.Linear(in_feature = embeding_size, out_feature = 3*attention_size, bias=False)

        if initialisation=="xavier":
            nn.init.xavier_normal_(self.w_qkv.weight)
        elif initialisation=="glorot":
            nn.init.xavier_uniform_(self.w_qkv.weight)
        elif initialisation=="he":
            nn.init.kaiming_normal_(self.w_qkv.weight)
        else:
            logger.error(f"Initialisation {initialisation} not supported. Please choose between 'xavier','glorot' and 'random'.")
            raise ValueError("Initialisation not supported. Please choose between 'xavier','glorot' and 'random'.")

    def forward(self, x: torch.tensor,mask:torch.tensor= None):
        # (batch, len, embedding size)
        x = self.w_qkv(x)
        q,k,v = x.chunk(3)
        
        scores = torch.matmul(q,k.transpose(1,2))
        if mask:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores)
        return torch.matmul(scores, v)
