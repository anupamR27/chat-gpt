import torch

torch.manual_seed(1337)

B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

print(x.shape)

xbow = torch.zeros((B, T, C))

for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)

print(xbow)

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)

xbow2 = wei @ x
print(xbow2)

print(torch.allclose(xbow, xbow2))