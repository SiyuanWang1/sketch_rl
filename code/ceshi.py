import torch

a=torch.arange(0,16).view(4,4)

b=torch.ones((4,4))
print(torch.nn.functional.mse_loss(a,b))

a=a.view(-1)
b=b.view(-1)
print(torch.nn.functional.mse_loss(a,b))