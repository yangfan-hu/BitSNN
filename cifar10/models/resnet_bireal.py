'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.binarized_modules import BinarizeConv2d


__all__ =['resnet18_bireal']

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.move0 = LearnableBias(in_planes)
        self.move1 = LearnableBias(planes)
        self.move2 = LearnableBias(planes)

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion*in_planes else planes // 4
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                        nn.AvgPool2d((2,2)), 
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))

    def forward(self, x):
        
        out = self.move0(x)
        
        out = F.hardtanh(self.bn1(self.conv1(out)), inplace=True)
        
        out = self.move1(out)
        
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.hardtanh(out, inplace=True)
        
        out = self.move2(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channel[0]

        self.conv1 = nn.Conv2d(3, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1 = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channel[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(num_channel[3]*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_channel[3]*block.expansion)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out 


def resnet18_bireal(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[64,128,256,512],**kwargs)




def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'linear' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
