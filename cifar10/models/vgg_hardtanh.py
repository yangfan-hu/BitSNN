'''VGG for CIFAR10.
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import math
# import torch.nn.init as init
from models.binarized_modules import BinarizeConv2d


__all__ = ['vgg_small_hardtanh']


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class VGG(nn.Module):

    def __init__(self, num_classes=10, sg='s_lste', min_val=-1, max_val=1, theta=0, delta=1, rho=0.3, phi=1):
        super(VGG, self).__init__()
        
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear0 = nn.Hardtanh(inplace=True)

        self.move1 = LearnableBias(128)
        self.conv1 = BinarizeConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        
        self.move2 = LearnableBias(128)
        self.conv2 = BinarizeConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        
        self.move3 = LearnableBias(256)
        self.conv3 = BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.nonlinear3 = nn.Hardtanh(inplace=True)
        
        self.move4 = LearnableBias(256)
        self.conv4 = BinarizeConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.nonlinear4 = nn.Hardtanh(inplace=True)
        
        self.move5 = LearnableBias(512)
        self.conv5 = BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.nonlinear5 = nn.Hardtanh(inplace=True)

        self.fc = nn.Linear(512*4*4, num_classes)        
        
        self._initialize_weights()
        
        # self.conv1.phi.data = self.conv1.phi.data*0+math.sqrt(2)
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear0(x)
        
        x = self.move1(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        
        x = self.move2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        
        x = self.move3(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear3(x)
        
        x = self.move4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear4(x)
        
        x = self.move5(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear5(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)        
        
        return x

def vgg_small_hardtanh(**kwargs):
    model = VGG(**kwargs)
    return model


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'fc' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('vgg'):
            print(net_name)
            test(globals()[net_name]())
            print()
    # a = vgg_small_1w1a(min_val=-6,max_val=6)