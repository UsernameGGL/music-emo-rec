import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
transform=transforms.Compose([transforms.Normalize([1],[1])])
a = torch.ones(1,3,3)
print(a)
c=transform(a)
print(c)


