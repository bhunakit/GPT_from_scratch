import os
import requests
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


input_file_path = os.path.join(os.path.dirname('__file__'), 'data/input.txt')

# download the tiny shakespeare dataset
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)

harry_potter_files = ['data/harry_potter1.txt', 'data/harry_potter2.txt', 'data/harry_potter3.txt', 'data/harry_potter4.txt']
with open(input_file_path, 'w') as outfile:
    for file in harry_potter_files:
        with open(file, 'r', encoding='utf-8', errors='replace') as infile:
            outfile.write(infile.read())
       

with open(input_file_path, 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

str_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_str = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [str_to_int[ch] for ch in s]
decode = lambda l: ''.join([int_to_str[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]


# global variables / hyperparameters
torch.manual_seed(1337)
batch_size = 8
block_size = 32
n_embd = 186
n_head = 3
n_layer = 2
dropout = 0.2

learning_rate = 3e-4
num_iterations = 5001
eval_interval = 501
eval_iterations = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def generate_batch(split, batch_size=4):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = generate_batch('train')

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # (batch size, block size, num embeddings)
        k = self.key(x) # (batch size, block size, head size)
        q = self.query(x) # (batch size, block size, head size)

        w = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (batch size, block size, block size)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)

        w = self.dropout(w)

        v = self.value(x) # (batch size, block size, head size)

        out = w @ v # (batch size, block size, head size)

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), 
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Transformer Block: intersperse self-attention and communication

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # divide head to multiple heads and concat back
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # self-attention transformation (B, T, C) --> (B, T, head_size(in this case: C))
        x = x + self.ffwd(self.ln2(x)) # feed foward one layer (B, T, head_size(in this case: C))

        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, idx):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape

        token_embd = self.token_embedding_table(idx) #(B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T)) #(T, C)
        x = token_embd + pos_embd #(B, T, C)

        x = self.block(x) 
        logits = self.lm_head(x) #(B, T, vocab_size)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target) 
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -block_size:]

            logits, loss = self.forward(idx_cropped)

            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def backward(self, batch_size=32):
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

        for step in range(num_iterations):
            xb, yb = generate_batch('train', batch_size)

            logits, loss = self.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if step % 1000 == 0:
                print(f"Step {step}: {loss.item()}")


m = BigramLanguageModel(xb)
m = m.to(device)

logits, loss = m.forward(xb, yb)

idx = torch.zeros((1, 1), dtype=torch.long, device=device)

print("Before Training:")
print(decode(m.generate(idx, max_new_tokens=400)[0].tolist()))

m.backward()

print("\nAfter Training:")
print(decode(m.generate(idx, max_new_tokens=400)[0].tolist()))


