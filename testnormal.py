
import torch
a = torch.Tensor([[1,2,3]])
b = torch.cat((a, torch.Tensor([[2, 3, 4]])))
print(b)
print(a)
