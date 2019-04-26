import torch
a = torch.Tensor([0, 0, 1, 2, 1])
print(sum(a[a>0]))
