import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

x = torch.rand(3, 3, device=device)

print("Device:", device)
print(x)