import torch
from torch.nn import functional as F
import torch.nn as nn

torch.manual_seed(1337)

B, T, C = 4, 8, 2

x = torch.randn(B, T, C)

head_size = 16

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)

print(x.shape)

xbow = torch.zeros((B, T, C))

for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)

print(xbow)

tril = torch.tril(torch.ones(T, T))

k = key(x)
q = query(x)

wei = q @ k.transpose(-2, -1)

wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

xbow2 = wei @ x

print(xbow2)