import os
import requests
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


torch.manual_seed(1337)
B, T, C = 4, 8, 16
x = torch.randn(B, T, C)

def self_attention_v1(x):
    B, T, C = x.shape

    tril = torch.tril(torch.ones((T, T)))
    w = torch.zeros((T, T))
    w = w.masked_fill(tril==0, float('-inf'))
    w = F.softmax(w, dim=1)

    xbow = w @ x

    return xbow

torch.manual_seed(1337)
B, T, C = 4, 8, 32 # example token_embedding + position_embedding
x = torch.randn(B, T, C) # represent token embedding

def self_attention_v2(x): #(ATTENTION IS ALL YOU NEED)
    B, T, C = x.shape

    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    k = key(x) # (B, T, 16)
    q = query(x) # (B, T, 16)

    w = q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, 16) @ (B, 16, T) --> (B, T, T)

    tril = torch.tril(torch.ones((T, T)))
    # w = torch.zeros((T, T))
    w = w.masked_fill(tril==0, float('-inf'))
    w = F.softmax(w, dim=1)

    v = value(x) # linear transform token embedding 

    xbow = w @ v

    return xbow
