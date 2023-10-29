import torch
import torch.nn as nn
from model import resnet50x1, resnet50x2, resnet50x4, MLP

checkpoint_1 = torch.load("resnet50-1x.pth")
checkpoint_2 = torch.load("resnet50-2x.pth")
checkpoint_4 = torch.load("resnet50-4x.pth")
resnet_1 = resnet50x1().to('cpu')
#resnet_2 = resnet50x2().to('cpu')
#resnet_4 = resnet50x4().to('cpu')
dsMLP = MLP(1500)
resnet_1.load_state_dict(checkpoint_1['state_dict'])

resnet_1.eval()
#resnet_2.eval()
#resnet_4.eval()
input = torch.randn(1, 3, 100, 100)
output = resnet_1(input)
output = dsMLP(output)
#output = resnet_2(input)
#output = resnet_4(input)
print('#######')