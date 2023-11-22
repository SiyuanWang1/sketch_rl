import torch


a=torch.arange(0,15).view(3,5)
print(a)
print(torch.sum(a))